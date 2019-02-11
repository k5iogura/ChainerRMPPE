#!/usr/bin/env python3
from pdb import *
import sys,os
from time import time
import numpy as np
import cv2
import argparse
from openvino.inference_engine import IENetwork, IEPlugin

def IEsetup(model_xml, model_bin, device, verbose=False):
    start =time()
    plugin = IEPlugin(device=device, plugin_dirs=None)
    libcpu = "inference_engine_samples/intel64/Release/lib/libcpu_extension.so"
    libcpu = os.environ['HOME'] + "/" + libcpu
    if device == "CPU":plugin.add_cpu_extension(libcpu)
    net = IENetwork(model=model_xml, weights=model_bin)

    if verbose:print("* IEsetup", model_bin, "on", device)
    exec_net = plugin.load(network=net, num_requests=1)

    input_blob = next(iter(net.inputs))  #input_blob = 'data'
    model_n, model_c, model_h, model_w = net.inputs[input_blob].shape
    if verbose:print("network in shape n/c/h/w (from xml)= %d %d %d %d"%(model_n, model_c, model_h, model_w))
    if verbose:print("input_blob =",input_blob)
    out_blobs = []
    for out_blob in net.outputs:
        if verbose:print("  net.outputs[",out_blob,"].shape",net.outputs[out_blob].shape)
        out_blobs.append(out_blob)
    if verbose:print("* IEsetup done %.2fmsec"%(1000.*(time()-start)))
    del net
    return exec_net, plugin, input_blob, out_blobs

def IEinfer(exec_net, in_frame, input_blob, out_blobs, verbose=False):
    if verbose:print("* IEinfer")
    start = time()
    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})
    result = {}
    if exec_net.requests[0].wait(-1) == 0:
        sec = time() - start
        for i, out_blob in enumerate(out_blobs):
            result[out_blob] = exec_net.requests[0].outputs[out_blob]
    else:
        print("error")
    if verbose:print("* IEinfer done %.2fmsec"%(1000.*(time()-start)))
    return result

def IEresult(_xml, _bin, _device, in_frame):
    exec_net, plugin, input_blob, out_blobs = IEsetup(_xml, _bin, _device)
    result = IEinfer(exec_net, in_frame, input_blob, out_blobs)
    return result

if __name__  == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("images", nargs='*', type=str)
    args.add_argument("-d", "--device"   , type=str, default="CPU", help="Default CPU")
    args.add_argument("-b", "--bin"      , type=str, required=True, help="IRmodel bin file")
    args.add_argument("-x", "--xml"      , type=str, required=True, help="IRmodel xml file")
    args = args.parse_args()

    data_type="FP16"
    if args.device == "CPU": data_type="FP32"
    assert os.path.exists(args.bin) 

    exec_net, plugin, input_blob, out_blobs = IEsetup(args.xml, args.bin, args.device, verbose=True)
    del exec_net
    del plugin

