import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
# print(parser.get_used_vc_plugin_libraries())
success = parser.parse_from_file('./backbone.onnx')
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    print('not success!')

config = builder.create_builder_config()
profile = builder.create_optimization_profile()
profile.set_shape('input', [1, 3, 320, 320], [
                  1, 3, 800, 1344], [1, 3, 1344, 1344])
config.add_optimization_profile(profile)
serialized_engine = builder.build_serialized_network(network, config)
with open("backbone.engine", "wb") as f:
    f.write(serialized_engine)
