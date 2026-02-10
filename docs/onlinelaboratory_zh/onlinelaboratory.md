# 线上实验室用户指南

1. 登录 Flag OS 后，点击右上角的 **线上实验室** 标签页。

2. 查看与你账号关联的所有未释放的环境容器、算力资源详情、访问入口及其他相关信息。  
   ![alt text](asset/online-lab.jpg)

3. 在 **环境访问** 列中，使用以下任一方式访问云端在线开发环境：
    - **方式一：直接访问开发环境**  
      如需直接访问开发环境，请按以下步骤操作：  
      1. 在 **密钥** 旁点击复制图标，复制密钥。  
      2. 点击 **进入IDE**。在弹出的 Welcome 对话框中粘贴密钥，并点击 **Submit**。  
      ![alt text](asset/welcome.jpg)
    - **方式二：通过公网访问**  
      如需通过公网访问开发环境，请按以下步骤操作：  
      1. 可将开发环境的服务映射到端口 30000。  
      2. 在 **密钥** 下方点击 **更多访问**。  
      3. 在弹出的 **更多访问信息** 对话框中，点击 **服务地址** 链接以打开开发环境。  
      ![alt text](asset/public-access.jpg)

4. 根据显卡类型，通过终端命令查询算力配置信息。  
    - 对于天数加速卡，使用以下命令：

       ```{code-block} bash
       ixsm
       ```

      ![alt text](asset/ascend-gpu-info.jpg)
    - 对于华为昇腾加速卡，使用以下命令：

       ```{code-block} python
       npu-smi info
       ```

      ![alt text](asset/iluvatar-gpu-info.jpg)

5. 您可以通过以下方式上传或下载代码包、模型等文件：
     - 在 `Workspace` 上点击右键，选择 **Upload...**  
      ![alt text](asset/upload.jpg)
     - 在 `Workspace` 上点击右键，选择 **Download...**  
      ![alt text](asset/download.jpg)

```{warning}
实验环境为容器化环境，在释放后所有数据将被永久删除且无法恢复，请务必提前在本地备份数据。
