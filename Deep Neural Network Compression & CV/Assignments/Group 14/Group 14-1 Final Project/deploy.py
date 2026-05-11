from deployment.load_model import load_compressed_model, run_inference

model = load_compressed_model("compressed_models/vgg_compressed.npz")
run_inference(model)