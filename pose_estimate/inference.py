#! /usr/bin/python3
import torch
import numpy as np
import torch.nn as nn
import cv2
import os
import argparse
import configparser
from Encoder import Encoder
from Model import Model

import rospy
#from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseArray
import tf
from geometry_msgs.msg import Pose
#from cv_bridge import CvBridge

from utils.plots import Annotator, colors

# TODO get some of these values from config instead
# NUM_VIEWS = 10
# NUM_OBJ = 30
DEBUG = True
SAVE_INTERVAL = 20

# local copy to avoid relying on utils due to name clash
def loadCheckpoint(model_path, encoder, args, num_obj):
    # Load checkpoint and parameters
    checkpoint = torch.load(model_path)

    # Load model
    #num_views = int(checkpoint['model']['l3.bias'].shape[0]/(6+1))
    num_views = args.get('Rendering', 'VIEWS', fallback=10)
    #model = Model(num_views=num_views).cuda()
    # model = Model(num_views=len(views),
    #               num_objects=len(model_path_loss),
    #               finetune_encoder=args.getboolean('Training','FINETUNE_ENCODER', fallback=False),
    #               classify_objects=args.getboolean('Training','CLASSIFY_OBJECTS', fallback=False),
    #               weight_init_name=args.get('Training', 'WEIGHT_INIT_NAME', fallback=""))
    model = Model(num_views=num_views,
                  num_objects=num_obj,
                  finetune_encoder=True,
                  classify_objects=False,
                  weight_init_name="")

    # TODO: These are probably just overwritten, check if so
    model.encoder.state_dict()['weight'].copy_(encoder.encoder_dense_MatMul.state_dict()['weight'])
    model.encoder.state_dict()['bias'].copy_(encoder.encoder_dense_MatMul.state_dict()['bias'])

    encoder.encoder_dense_MatMul = None

    model.load_state_dict(checkpoint['model'])

    print("Loaded the checkpoint: \n" + model_path)
    return model, num_views

# batch*n
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3

    return out

# again copy to avoid importing utils
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def rtToMat(rot, t):
    mat = np.concatenate((rot, np.array(t).reshape(3,1)), axis=1)
    mat = np.concatenate((mat, [[0, 0, 0, 1]]), axis=0)
    return mat

def conv_R_opengl2pytorch_np(R):
    # Convert R matrix from opengl to pytorch format
    xy_flip = np.eye(3, dtype=np.float)
    xy_flip[0,0] = -1.0
    xy_flip[1,1] = -1.0
    R_pytorch = np.transpose(R)
    R_pytorch = np.dot(R_pytorch,xy_flip)

    # Convert to tensors
    #R = torch.from_numpy(R_conv)
    return R_pytorch

def conv_R_pytorch2opengl_np(R):
    # Convert R matrix from pytorch to opengl format
    xy_flip = np.eye(3, dtype=np.float)
    xy_flip[0,0] = -1.0
    xy_flip[1,1] = -1.0
    R_opengl = np.dot(R,xy_flip)
    R_opengl = np.transpose(R_opengl)

    return R_opengl

def correct_R(R, t_est):
    # correcting the rotation matrix
    # the codebook consists of centered object views, but the test image crop is not centered
    # we determine the rotation that preserves appearance when translating the object

    # SHBE fix - x and y translation should be negative/inverted like opengl2pytorch conversion?
    t_est = t_est*np.array([-1.0,-1.0,1.0])

    d_alpha_y = np.arctan(t_est[0]/np.sqrt(t_est[2]**2+t_est[1]**2))
    d_alpha_x = - np.arctan(t_est[1]/t_est[2])
    R_corr_x = np.array([[1,0,0],
                         [0,np.cos(d_alpha_x),-np.sin(d_alpha_x)],
                         [0,np.sin(d_alpha_x),np.cos(d_alpha_x)]])
    R_corr_y = np.array([[np.cos(d_alpha_y),0,np.sin(d_alpha_y)],
                         [0,1,0],
                         [-np.sin(d_alpha_y),0,np.cos(d_alpha_y)]])
    R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,R))
    return R_corrected

class Inference():
    def __init__(self, args, num_obj):
        self.args = args
        self.num_obj = num_obj
        # TODO get these paths and values from config instead
        detector_weight_path = "./weights/detector.pt"
        detector_repo = "./"
        encoder_weights = "./weights/encoder.npy"
        pose_estimator_weights = "./weights/pose_estimator.pt"

        device = torch.device("cuda:0")

        # load yolo detector
        detector = torch.hub.load(detector_repo, 'custom', path=detector_weight_path, source='local')
        detector.conf = 0.20 # change confidence threshold if necessary

        # load AE autoencoder
        encoder = Encoder(encoder_weights).to(device)
        encoder.eval()

        # load pose estimator
        model, num_views = loadCheckpoint(pose_estimator_weights, encoder, self.args, num_obj)
        model.to(device)
        model = model.eval() # Set model to eval mode

        self.device = device
        self.detector = detector
        self.encoder = encoder
        self.model = model
        self.num_views = num_views

    def get_rotpred_for_obj(self, predicted_poses, obj_id):
        views = self.num_views
        # confidences ordered by object first, then views
        confs = predicted_poses[:,views * obj_id:views * (obj_id + 1)]
        # which pose has highest conf
        index = torch.argmax(confs)
        if DEBUG:
            print(index.tolist())
            print(confs.tolist())
            print(np.sum(confs.tolist()))
        # jump past confidences and to the right object and right view
        pose_start = self.num_obj * self.num_views + obj_id * self.num_views * 6 +  index * 6 # +3 if with translation
        pose_end = pose_start + 6
        curr_pose = predicted_poses[:,pose_start:pose_end]
        Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
        return Rs_predicted.cpu().detach().numpy()[0]

    def mid_depth(self, detection, points_data):
        box = detection['box']
        tlx = box[0].cpu()
        tly = box[1].cpu()
        brx = box[2].cpu()
        bry = box[3].cpu()

        bbox = [tlx, tly, brx, bry]

        mid_x = int((tlx + brx)/2)
        mid_y = int((tly + bry)/2)

        if points_data is None:
            return None, bbox

        temp = point_cloud2.read_points(points_data, field_names=['x', 'y', 'z'], skip_nans=True, uvs=[(mid_x, mid_y)])
        T = list(list(temp)[0])

        # throw away if no good depth for now TODO
        if T[2] < 0.01:
            print(f"T of {T} thrown away")
            return None, bbox

        return T, bbox

    def process_crop(self, detection):
        # Disable gradients for the encoder
        with torch.no_grad():
            # detect object in scene, get bboxes, crop and pass to AE for each object
            #pred = self.detector.predict(scene_image)
            #print(type(scene_image))
            #print(scene_image.shape)
            #pred = self.detector.run([self.detector.get_outputs()[0].name], {self.detector.get_inputs()[0].name: scene_image.astype(np.float32)})[0]
            #pred = detector(scene_img)

            resize=(128,128)
            interpolation=cv2.INTER_NEAREST

            # comments are approximate tensor format
            image = detection['im']
            # box = detection['box'] # ex [476.99997, 112.96561, 848., 480.]
            # conf = detection['conf'] # ex 0.32861
            cls = detection['cls'] # ex 7.
            # label = detection['label'] # ex '8 0.33'

            img = cv2.resize(image, resize, interpolation = interpolation)
            if DEBUG:
                cv2.imwrite("./debug_output/post_resize_crop_cls_{}.png".format(cls), img)

            # Trying some color changes to see if that is the issue
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Normalize image
            img_max = np.max(img)
            img_min = np.min(img)
            img = img.astype('float32')
            img = (img - img_min)/(img_max - img_min)

            # Run image through encoder
            img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(self.device)
            if DEBUG:
                print(type(img), img.shape)
            #print(img.shape)
            code = self.encoder(img)
            if DEBUG:
                print("encoder output shape {}".format(code.shape))
            code = code.detach().cpu().numpy()[0]

        # Predict poses from the codes
        batch_codes = torch.tensor(np.stack([code]), device=self.device, dtype=torch.float32)
        predicted_poses = self.model(batch_codes)

        rot = self.get_rotpred_for_obj(predicted_poses, cls.type(torch.int64))

        return rot

    def process_scene(self, image, points_data):
        detections = self.detector(image)

        crops_det = detections.crop(save=False) # set to True to debug crops, but is broken
        ret = []
        line_thickness = 3
        annotator = Annotator(image.copy(), line_width=line_thickness) # example=str(names)
        for i, crop_det in enumerate(crops_det):
            T, bbox = self.mid_depth(crop_det, points_data)
            if not T:
                continue
            Rs_predicted = self.process_crop(crop_det)
            crop_det['rot'] = Rs_predicted
            crop_det['T'] = T
            crop_det['bbox'] = bbox
            ret.append(crop_det)

            # debugging and such
            if DEBUG:
                print()
                print(crop_det['cls'])
                print(crop_det['label'])

                annotator.box_label(crop_det['bbox'], crop_det['label'], color=colors(crop_det['cls'], True))

                # TODO save crops to file too
                cv2.imwrite("./debug_output/crop{}.png".format(i), crop_det['im'])

        if DEBUG:
            yoloimg = annotator.result()
        else:
            yoloimg = None
        return ret, yoloimg

class Pose_estimation_rosnode():
    def __init__(self, args, num_obj):
        self.args = args
        self.num_obj = num_obj
        self.inference = Inference(args, self.num_obj)
        self.depth = None
        self.points_data = None
        #self.br = CvBridge()
        self.pub = rospy.Publisher('pose_estimation', PoseArray, queue_size=10)
        self.pubYolo = rospy.Publisher('yolo_detections', Image, queue_size=10)
        self.debug_counter = 0
        rospy.init_node('pose_estimation_hampus', anonymous=True)
        #rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback=self.depth_callback)
        #rospy.Subscriber('/camera/color/image_raw', Image, callback=self.run_callback)
        #rospy.Subscriber('/camera/depth/color/points', PointCloud2, callback=self.points_callback)
        rospy.Subscriber('/realsense/aligned_depth_to_color/image_raw', Image, callback=self.depth_callback)
        rospy.Subscriber('/realsense/rgb/image_raw', Image, callback=self.run_callback)
        rospy.Subscriber('/realsense/depth/points', PointCloud2, callback=self.points_callback)

        rospy.spin()

    def depth_callback(self, depth_data):
        rospy.loginfo("depth_callback called")
        #self.depth = self.bridge.imgmsg_to_cv2(data, 'passthrough')
        self.depth = np.frombuffer(depth_data.data, dtype=np.uint8).reshape(depth_data.height, depth_data.width, -1)

    def points_callback(self, point_data):
        rospy.loginfo("points_callback called")
        self.points_data = point_data

    def run_callback(self, image_data):
        rospy.loginfo("run_callback called")
        #image = self.bridge.imgmsg_to_cv2(data, 'passthrough')
        image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        
        if False: # testing override image
            image = cv2.imread("./testing/10-0114.png")
            #image = cv2.imread("./testing/10-0492.png")
            #image = cv2.imread("./testing/8-0102.png")
            #image = cv2.imread("./testing/8-0232.png")
            image_data.width = 720
            image_data.height = 540

        pred, yoloimg = self.inference.process_scene(image, self.points_data)
        #rospy.loginfo(pred)

        msg = PoseArray()
        msg.header = image_data.header # TODO this might be wrong, but lets
        for i, p in enumerate(pred):
            T = p['T']

            # throw away if no good depth for now TODO
            if not T or T[2] < 0.01:
                continue

            #p['rot'] = conv_R_pytorch2opengl_np(p['rot'])
            # correct rotation for location in image
            p['rot'] = correct_R(p['rot'], T)
            # convert to opengl (cameracentric I think) format
            p['rot'] = conv_R_pytorch2opengl_np(p['rot'])
            #p['rot'] = conv_R_opengl2pytorch_np(p['rot'])

            pose = Pose()
            pose.position.x = T[0]
            pose.position.y = T[1]
            pose.position.z = T[2]

            mat = rtToMat(p['rot'], T)
            #print(mat)
            qt = tf.transformations.quaternion_from_matrix(mat)
            pose.orientation.x = qt[0]
            pose.orientation.y = qt[1]
            pose.orientation.z = qt[2]
            pose.orientation.w = qt[3]
            msg.poses.append(pose)

            if DEBUG and self.debug_counter % SAVE_INTERVAL == 0:
                ind = self.debug_counter/SAVE_INTERVAL
                with open("./debug_output/RnT_{}_{}.txt".format(ind, i), "w") as f:
                        f.write("{} \n{}".format(p['rot'], T))

        if DEBUG:
            yolomsg = Image() #= self.br.cv2_to_imgmsg(annotator.result(), 'passthrough')
            yolomsg.header = image_data.header # TODO this might be wrong, but lets
            yolomsg.height = image_data.height
            yolomsg.width = image_data.width
            yolomsg.encoding = image_data.encoding
            yolomsg.is_bigendian = image_data.is_bigendian
            yolomsg.step = image_data.step
            yolomsg.data = yoloimg.flatten().tobytes()
            self.pubYolo.publish(yolomsg)
            self.pub.publish()

            if len(pred)>0:
                ind = self.debug_counter/SAVE_INTERVAL
                cv2.imwrite("./debug_output/rgb_image_{}.png".format(ind), image)
                cv2.imwrite("./debug_output/rgb_image_{}_annotated.png".format(ind), yoloimg)

                self.debug_counter += 1

        self.pub.publish(msg)

# read config and spin up rosnode
if __name__ == '__main__':
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    arguments = parser.parse_args()

    cfg_file_path = os.path.join("./", arguments.config)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    objs = args.get('Dataset', 'MODEL_PATH_DATA', fallback=None)
    if objs:
        num_obj = len(objs)
    else:
        num_obj = 30

    pose_estimation_rosnode = Pose_estimation_rosnode(args, num_obj)
