import onnx

onnx_model = onnx.load('end2end.onnx')
graph = onnx_model.graph
node = graph.node
initializers = graph.initializer
k0 = onnx.helper.make_tensor(
    name='TopK_0_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[100])
k1 = onnx.helper.make_tensor(
    name='TopK_1_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[100])
k2 = onnx.helper.make_tensor(
    name='TopK_2_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[100])
k3 = onnx.helper.make_tensor(
    name='TopK_3_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[100])
k4 = onnx.helper.make_tensor(
    name='TopK_4_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[100])
k5 = onnx.helper.make_tensor(
    name='TopK_5_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[100])
k6 = onnx.helper.make_tensor(
    name='TopK_6_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[50])
k7 = onnx.helper.make_tensor(
    name='TopK_7_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[50])
k8 = onnx.helper.make_tensor(
    name='TopK_8_k', data_type=onnx.TensorProto.INT64, dims=[1], vals=[30])
graph.initializer.append(k0)
graph.initializer.append(k1)
graph.initializer.append(k2)
graph.initializer.append(k3)
graph.initializer.append(k4)
graph.initializer.append(k5)
graph.initializer.append(k6)
graph.initializer.append(k7)
graph.initializer.append(k8)

for i in range(len(node)):
    op = node[i]
    if op.name == '/TopK':
        print(op)
        op.input[1] = 'TopK_0_k'
    if op.name == '/TopK_1':
        print(op)
        op.input[1] = 'TopK_1_k'
    if op.name == '/TopK_2':
        print(op)
        op.input[1] = 'TopK_2_k'
    if op.name == '/TopK_3':
        print(op)
        op.input[1] = 'TopK_3_k'
    if op.name == '/TopK_4':
        print(op)
        op.input[1] = 'TopK_4_k'
    if op.name == '/TopK_5':
        print(op)
        op.input[1] = 'TopK_5_k'
    if op.name == '/TopK_6':
        print(op)
        op.input[1] = 'TopK_6_k'
    if op.name == '/TopK_7':
        print(op)
        op.input[1] = 'TopK_7_k'
    if op.name == '/TopK_8':
        print(op)
        op.input[1] = 'TopK_8_k'

# graph = onnx.helper.make_graph(
#     graph.node, graph.name, graph.input, graph.output, graph.initializer)
# info_model = onnx.helper.make_model(graph)
# onnx_model = onnx.shape_inference.infer_shapes(info_model)

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'end2end_change.onnx')
