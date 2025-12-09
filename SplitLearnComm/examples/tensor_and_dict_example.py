"""
一个最小可运行示例，展示 splitlearn-comm 传输张量，以及用“张量包装”方式传字典。

用法：
  # 终端 1：启动服务端（张量模式）
  python examples/tensor_and_dict_example.py --role server --mode tensor --port 50052
  # 终端 2：客户端调用
  python examples/tensor_and_dict_example.py --role client --mode tensor --port 50052

  # 服务端处理“字典”示例（将字典转成张量传输）
  python examples/tensor_and_dict_example.py --role server --mode dict --port 50053
  python examples/tensor_and_dict_example.py --role client --mode dict --port 50053

说明：
- 通信协议原生只支持单个张量，本示例的“字典传输”是通过
  JSON -> bytes -> torch.float32 张量 -> 传输 -> 还原，属于演示用的包装方案。
- 若需正式支持多张量/字典，推荐在协议层或 TensorCodec 做扩展。
"""

import argparse
import json
import time
from typing import Dict, Any

import torch

from splitlearn_comm import GRPCComputeServer, GRPCComputeClient
from splitlearn_comm.core import ComputeFunction, ModelComputeFunction


# ----------------------- 工具：字典 <-> 张量（浮点包装） ----------------------- #
def dict_to_tensor(payload: Dict[str, Any]) -> torch.Tensor:
    """
    将字典序列化为 JSON，再把 bytes 写入 float32 张量。
    注意：仅用于演示；真实场景应在协议或编码器层扩展。
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    arr = torch.tensor(list(data), dtype=torch.float32)
    return arr


def tensor_to_dict(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    将 float32 张量还原为 JSON 字典。
    """
    byte_list = tensor.to(torch.uint8).cpu().numpy().tolist()
    data = bytes(byte_list)
    return json.loads(data.decode("utf-8"))


# ----------------------- ComputeFunction 实现 ----------------------- #
class DictEchoComputeFunction(ComputeFunction):
    """
    接收“字典包装成张量”，返回附加信息后的字典（同样包装成张量）。
    """

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        payload = tensor_to_dict(input_tensor)
        payload["server_time"] = time.time()
        payload["received_len"] = len(payload)
        return dict_to_tensor(payload)

    def get_info(self):
        return {"name": "DictEcho", "device": "cpu"}


def build_tensor_server(port: int):
    # 一个简单线性模型，演示张量前向
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )
    compute_fn = ModelComputeFunction(model, device="cpu", model_name="demo-linear")
    server = GRPCComputeServer(compute_fn=compute_fn, host="0.0.0.0", port=port)
    return server


def build_dict_server(port: int):
    compute_fn = DictEchoComputeFunction()
    server = GRPCComputeServer(compute_fn=compute_fn, host="0.0.0.0", port=port)
    return server


# ----------------------- 客户端调用 ----------------------- #
def run_tensor_client(port: int):
    client = GRPCComputeClient(f"localhost:{port}")
    client.connect()
    x = torch.randn(2, 4)
    y = client.compute(x)
    print(f"[tensor] input shape={tuple(x.shape)}, output shape={tuple(y.shape)}")
    client.close()


def run_dict_client(port: int):
    client = GRPCComputeClient(f"localhost:{port}")
    client.connect()
    payload = {"msg": "hello", "step": 1}
    encoded = dict_to_tensor(payload)
    resp_tensor = client.compute(encoded)
    resp = tensor_to_dict(resp_tensor)
    print(f"[dict] send={payload} -> recv={resp}")
    client.close()


# ----------------------- CLI ----------------------- #
def main():
    parser = argparse.ArgumentParser(description="splitlearn-comm tensor & dict demo")
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--mode", choices=["tensor", "dict"], required=True)
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()

    if args.role == "server":
        server = build_tensor_server(args.port) if args.mode == "tensor" else build_dict_server(args.port)
        server.start()
        print(f"[server] mode={args.mode}, port={args.port}")
        server.wait_for_termination()
    else:
        if args.mode == "tensor":
            run_tensor_client(args.port)
        else:
            run_dict_client(args.port)


if __name__ == "__main__":
    main()
