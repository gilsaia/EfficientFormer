import tensorrt as trt
from mmdet.models.builder import BACKBONES as det_BACKBONES
import torch
import torch.nn as nn
import numpy as np
from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin

load_tensorrt_plugin()


class OwnAllocator(trt.IOutputAllocator):
    def __init__(self, shape, type=torch.float32):
        super().__init__()
        self.output = torch.empty(
            shape, device=f'cuda:{torch.cuda.current_device()}', dtype=type)

    def notify_shape(self, tensor_name, shape):
        pass

    def reallocate_output(self, tensor_name, memory, size, alignment):
        return self.output.data_ptr()

    def get_torch_tensor(self):
        return self.output


class EfficientFormerTrt(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open('./deploy_res/end2end.engine', 'rb') as f:
            serialized_engine = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.context.set_output_allocator(
            'dets', OwnAllocator([1*200*5]))
        self.context.set_output_allocator(
            'labels', OwnAllocator([1*200], torch.int32))
        # self.context.set_output_allocator(
        #     'y.72', OwnAllocator([1*200*28*28]))
        self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        # self.output1 = cuda.mem_alloc(1*32*336*336*4)
        # self.output2 = cuda.mem_alloc(1*64*168*168*4)
        # self.output3 = cuda.mem_alloc(1*144*84*84*4)
        # self.output4 = cuda.mem_alloc(1*288*42*42*4)
        # self.context.set_tensor_address('input.124', self.output1)
        # self.context.set_tensor_address('input.244', self.output2)
        # self.context.set_tensor_address('input.796', self.output3)
        # self.context.set_tensor_address('2892', self.output4)

    def forward(self, x):
        input_shape = list(x.shape)
        self.context.set_tensor_address('input', x.data_ptr())
        self.context.set_input_shape('input', input_shape)
        self.context.infer_shapes()
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        output1_shape = list(self.context.get_tensor_shape('dets'))
        output2_shape = list(self.context.get_tensor_shape('labels'))
        # output3_shape = list(self.context.get_tensor_shape('y.72'))
        output1_size = output1_shape[0]*output1_shape[1] * \
            output1_shape[2]
        output2_size = output2_shape[0]*output2_shape[1]
        # output3_size = output3_shape[0]*output3_shape[1] * \
        #     output3_shape[2]*output3_shape[3]
        output1_al = self.context.get_output_allocator('dets')
        output2_al = self.context.get_output_allocator('labels')
        # output3_al = self.context.get_output_allocator('y.72')
        output1 = output1_al.get_torch_tensor(
        )[0:output1_size].view(output1_shape)
        output2 = output2_al.get_torch_tensor(
        )[0:output2_size].view(output2_shape)
        # output3 = output3_al.get_torch_tensor(
        # )[0:output3_size].view(output3_shape)
        output = [output1, output2]
        return output


@det_BACKBONES.register_module()
class efficientformerv2_s2_feat_trt(EfficientFormerTrt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
