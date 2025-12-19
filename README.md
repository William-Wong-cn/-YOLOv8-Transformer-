### 项目经验

**YOLOv8-Transformer 实时人体动作识别系统**  

- 基于 Ultralytics YOLOv8 实现多人实时人体检测，结合自研 Transformer 时序模型对连续帧序列进行动作分类，支持 6 类动作识别（jump / fall / near-fall / sitting / standing / running）  
- 实现端到端视频实时处理流程：在检测框上动态显示动作标签与置信度，支持文件夹监控，自动处理新放入的视频文件  
- 每视频处理完成后自动统计各类动作平均置信度与出现次数，并生成高清柱状图进行可视化分析（含保存功能）  
- 系统具备强扩展性与落地潜力，可直接应用于智慧养老（跌倒检测）、智能安防（异常行为预警）、体育训练分析等场景  
- 技术栈：Python、PyTorch、Ultralytics YOLOv8、OpenCV、Matplotlib、Torchvision  

项目亮点：完整实现了从检测 → 时序建模 → 可视化统计的全链路闭环，支持实时推理与批量处理，已在多段测试视频上验证有效性。
