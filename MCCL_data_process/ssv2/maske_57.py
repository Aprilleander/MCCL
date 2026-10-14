import random

def generate_subsets(input_file, output_70, output_50):
    # 读取所有行数据
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # 打乱数据顺序，确保随机性
    random.shuffle(lines)

    # 计算数量
    total_count = len(lines)
    count_70 = int(total_count * 0.7)
    count_50 = int(count_70 * 50/70)

    # 生成70%的数据
    subset_70 = lines[:count_70]

    # 从70%的数据中选择50%的数据
    subset_50 = random.sample(subset_70, count_50)

    # 写入70%的数据文件
    with open(output_70, 'w') as file_70:
        file_70.writelines(subset_70)

    # 写入50%的数据文件
    with open(output_50, 'w') as file_50:
        file_50.writelines(subset_50)

    print(f"文件生成完成：{output_70} 包含 {count_70} 条数据, {output_50} 包含 {count_50} 条数据")

# 使用示例
input_txt = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v70/label.txt'
t70 = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v70/label70.txt'
t50 = '/mnt/newdisk/mnt/newdisk/dataset/action/ssv2_all/label/initial/v70/label50.txt'
generate_subsets(input_txt, t70, t50)