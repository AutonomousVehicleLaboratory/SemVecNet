"""
python hrnet_semantic_segmentation.py --num_workers 1 --dataset=mapillary --cv=0 --bs_val=1 --eval=folder --eval_folder='./imgs/test_imgs' --dump_assets --dump_all_images --n_scales="0.5,1.0,2.0" --snapshot="ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth" --arch=ocrnet.HRNet_Mscale --result_dir=LOGDIR

Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
from PIL import Image
import argparse
import semantic_mapping.src.utils.mapillary_visualization as mapillary_visl

TRT_LOGGER = trt.Logger()


def get_custom_hrnet_args(cfg):
    def_args = argparse.Namespace()
    def_args.engine_file_path = cfg.VISION_SEM_SEG.ENGINE_FILE
    def_args.dummy_image_path = cfg.VISION_SEM_SEG.DUMMY_IMAGE_PATH

    return def_args


class HRNetSemanticSegmentationTensorRT():
    def __init__(self, args):
        """

        Args:
            args: engine file path
        """
        self.engine = self.load_engine(args.engine_file_path)
        dummy_input_file = args.dummy_image_path
        input_image, self.image_width, self.image_height = self.load_image(dummy_input_file)
        self.context = self.engine.create_execution_context()
        ### newer version (4090)
        # self.context.set_input_shape("x.1", (1, 3, self.image_height, self.image_width))
        ### older version (3070)
        self.context.set_binding_shape(self.engine.get_binding_index("x.1"), (1, 3, self.image_height, self.image_width))
        self.bindings = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                self.input_memory = cuda.mem_alloc(input_image.nbytes) # This can be done only once
                self.bindings.append(int(self.input_memory)) # This can be done only once
            else:
                self.output_buffer = cuda.pagelocked_empty(size, dtype) # This can be done only once
                self.output_memory = cuda.mem_alloc(self.output_buffer.nbytes) # This can be done only once
                self.bindings.append(int(self.output_memory)) # This can be done only once
        self.stream = cuda.Stream()
        # print(self.bindings)

        self.segmentation(input_image)
        self.segmentation(input_image)
        self.segmentation(input_image)


    def load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


    def load_image(self, input_file):
        with Image.open(input_file) as img:
            img = img.resize((1920 // 2, 1440 // 2))
            input_image = self.preprocess(np.array(img))
            image_width = img.width
            image_height = img.height
            return input_image, image_width, image_height


    def convert_cv2_PIL(self, img):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((1920 // 2, 1440 // 2))
        return img_pil
    

    def preprocess(self, image, network_image_shape = (1920 // 2, 1440 // 2)):
        # Mean normalization
        mean = np.array([0.485, 0.456, 0.406]).astype('float32')
        stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
        image_resized = cv2.resize(image, network_image_shape, interpolation=cv2.INTER_AREA)
        data = (np.asarray(image_resized).astype('float32') / float(255.0) - mean) / stddev
        # Switch from HWC to to CHW order
        return np.moveaxis(data, 2, 0)


    # def faster_postprocess(self, class_num, output_file):
    #     reshaped_output = np.reshape(self.output_buffer, (1, 1440 // 2, 1920 // 2))
    #     # reshaped_output_max = np.argmax(reshaped_output, axis = 0)
    #     reshaped_output_max = reshaped_output.squeeze()
    #     img = self.postprocess_map(reshaped_output_max)
    #     # print("Writing output image to file {}".format(output_file))
    #     img.convert('RGB').resize((1920, 1440), Image.NEAREST).save("/home/semantic_mapping/catkin_ws/test-3.png")


    # def postprocess_map(self, data):
    #     seg_color_ref = mapillary_visl.get_labels("/home/semantic_mapping/catkin_ws/src/online_semantic_mapping/config/config_65.json")
    #     colored_output = mapillary_visl.apply_color_map(data, seg_color_ref)
    #     img = Image.fromarray(colored_output.astype('uint8'))
    #     return img
    
    
    def segmentation(self, img):
        """
        Run semantic segmentation on a single numpy image in form
        (h, w, 3) where it is uint8 0-255
        """
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("Input Image")
        # print(img)
        # self.context = self.engine.create_execution_context()
        # self.context.set_input_shape("input.1", (1, 3, self.image_height, self.image_width))
        # self.stream = cuda.Stream()
        input_buffer = np.ascontiguousarray(img)
        cuda.memcpy_htod_async(self.input_memory, input_buffer, self.stream)
        # Run inference
        # self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.context.execute_v2(bindings=self.bindings)

        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(self.output_buffer, self.output_memory, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # print("Output Image")
        # print(self.output_buffer)
        # print("New input")
        # print(input_buffer)
        # reshaped_output = np.reshape(self.output_buffer, (65, 1440 // 2, 1920 // 2))
        # reshaped_output_max = np.argmax(reshaped_output, axis = 0)
        # self.faster_postprocess(65, "/home/semantic_mapping/catkin_ws/test-2.png")
        return self.output_buffer # reshaped_output_max


def main():
    print("Running TensorRT inference for HRNet")
    engine_file = "/home/semantic_mapping/catkin_ws/src/online_semantic_mapping/src/hrnet/assets/seg_weights/hrnet-avl-map.engine"
    folder_name = "/home/semantic_mapping/avt_cameras_camera1_image_rect_color_compressed"
    out_folder_name = "/home/semantic_mapping/avt_cameras_camera1_image_rect_color_compressed_seg_map"
    input_file  = "/home/tensorrt-dep/semantic-segmentation/imgs/test_imgs/nyc.jpg"
    output_file = "output.jpg"
    segmentation_model = HRNetSemanticSegmentationTensorRT(get_custom_hrnet_args())
    file = os.listdir(folder_name)[10]
    # for file in tqdm(os.listdir(folder_name)):
    # file = "/home/semantic_mapping/catkin_ws/src/online_semantic_mapping/src/hrnet/assets/seg_weights/1682354962.572154522.jpg"
    input_file = os.path.join(folder_name, file)
    input_image, _, _ = segmentation_model.load_image(input_file)
    segmentation_model.segmentation(input_image)
    segmentation_model.segmentation(input_image)
    output_file = os.path.join(out_folder_name, file)
    # segmentation_model.faster_postprocess(65, output_file)
    

if __name__ == '__main__':
    main()