import PySimpleGUI as sg
def get_siber():
    siber = [
        [sg.B("数据处理")],
        [sg.B("模型训练")],
        [sg.B("模型测试")]
    ]
    return siber

def data_process():
    siber = get_siber()
    fileBrower = [
        [sg.T("数据集-True"), sg.In(key="true_path"), sg.FileBrowse()],
        [sg.T("数据集-False"), sg.In(key="fake_path"), sg.FileBrowse()],
        [sg.T("词向量"), sg.In(key="word_model_path"), sg.FileBrowse()],
        [sg.T("参数设置")],
        [sg.T("train_size："), sg.In(key="train_size")],
        [sg.T("test_size："), sg.In(key="test_size")],
        [sg.T("random_state："), sg.In(key="random_state")],
    ]
    output = [
        [sg.Output(size=(30,20))],
        [sg.B("处理数据",key="处理数据"), sg.B("分割数据",key="分割数据")]
    ]
    layout = [
        [sg.Col(siber), sg.Col(fileBrower), sg.Col(output)]
    ]
    return layout

# data_process()
def model_train():
    siber = get_siber()
    input = [
        [sg.T("参数设置")],
        [sg.T("epochs："), sg.In(key='epochs', default_text=2)],
        [sg.T("batch_size："), sg.InputText(key='batch_size', default_text=64)],
        [sg.T("lr："), sg.In(key='lr',default_text=0.001)],
        [sg.T("output_size："), sg.In(key='output_size', default_text=1)],
        [sg.T("num_filters："), sg.In(key='num_filters', default_text=100)],
        [sg.T("kernel_sizes："), sg.In(key='kernel_sizes', default_text=[3,4,5])],
        [sg.T("dropout："), sg.In(key='dropout', default_text=0.5)],
    ]
    output = [
        #[sg.Output(size=(30, 20))],
        [sg.B("开始训练", key="开始训练")]
    ]
    layout = [
        [sg.Col(siber), sg.Col(input), sg.Col(output)]
    ]

    return layout
