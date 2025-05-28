#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from typing import List, Dict, Any

def load_and_merge_results(
    result_path: str,
    t_result_path: str,
    s_result_path: str,
    output_path: str = "new_result.pth",
    local_rank: int = 0
) -> Dict[str, Any]:
    
    # 初始化设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 加载所有结果文件
    print(f"Loading original results from {result_path}...")
    results = torch.load(result_path, map_location=device)
    
    print(f"Loading T-type results from {t_result_path}...")
    t_results = torch.load(t_result_path, map_location=device)
    
    print(f"Loading S-type results from {s_result_path}...")
    s_results = torch.load(s_result_path, map_location=device)

    # 验证必要字段存在
    required_fields = ["prototype_higher", "preds_higher"]
    for name, data in zip(["results", "t_results", "s_results"], 
                         [results, s_results, t_results]):
        if not all(field in data for field in required_fields):
            raise KeyError(f"{name} missing required fields")

    # 定义设备转移函数
    def move_to_device(x: Any) -> Any:
        """将张量移动到指定设备，其他类型保持不变"""
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, list):
            return [move_to_device(item) for item in x]
        return x

    # 提取并处理S型结果
    print("Processing S-type results...")
    s_prototype = move_to_device(s_results["prototype_higher"])
    s_preds = move_to_device(s_results["preds_higher"])

    # 提取并处理T型结果
    print("Processing T-type results...")
    t_prototype = move_to_device(t_results["prototype_higher"])
    t_preds = move_to_device(t_results["preds_higher"])

    # 合并prototype_higher层
    print("Merging prototype layers...")
    results["prototype_higher"] = [
        results["prototype_higher"][0],  # 原始第一层
        s_prototype[0],                   # 新增第二层（S型第一层）
        t_prototype[0],                   # 新增第三层（T型第一层）
        *results["prototype_higher"][1:]  # 原始后续层后移
    ]

    # 合并preds_higher层
    print("Merging prediction layers...")
    results["preds_higher"] = [
        results["preds_higher"][0],       # 原始第一层
        s_preds[0],                        # 新增第二层（S型第一层）
        t_preds[0],                        # 新增第三层（T型第一层）
        *results["preds_higher"][1:]       # 原始后续层后移
    ]

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存合并结果
    print(f"Saving merged results to {output_path}...")
    torch.save(results, output_path)
    print("Merge completed successfully!")
    return results

def main():
    parser = argparse.ArgumentParser(description="Merge multiple result files")
    parser.add_argument("--result", type=str, default="/mnt/weight/kmeans/ssv2_r_kmeans.pth",
                        help="Path to original results file")
    parser.add_argument("--t_result", type=str, default="/mnt/weight/kmeans/ssv2_t_kmeans.pth",
                        help="Path to T-type results file")
    parser.add_argument("--s_result", type=str, default="/mnt/weight/kmeans/ssv2_s_kmeans.pth",
                        help="Path to S-type results file")
    parser.add_argument("--output", type=str, default="/mnt/weight/kmeans/ssv2_kmeans.pth",
                        help="Output file path")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="CUDA device number for tensor operations")
    
    args = parser.parse_args()

    # 检查文件存在性
    for path in [args.result, args.t_result, args.s_result]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # 执行合并操作
    load_and_merge_results(
        result_path=args.result,
        t_result_path=args.t_result,
        s_result_path=args.s_result,
        output_path=args.output,
        local_rank=args.local_rank
    )

if __name__ == "__main__":
    main()