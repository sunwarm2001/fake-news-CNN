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
        [sg.T("数据集-True"), sg.In(key="true_data"), sg.FileBrowse()],
        [sg.T("数据集-False"), sg.In(key="fake_data"), sg.FileBrowse()],
        [sg.T("词向量"), sg.In(key="word_model"), sg.FileBrowse()],
    ]
    output = [
        [sg.Output(size=(30,20))],
        [sg.B("处理数据",key="处理数据")]
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
        [sg.T("epochs："), sg.In(key='epochs')],
        [sg.T("batch_size："), sg.InputText(key='batch_size', default_text=64)],
        [sg.T("lr"), sg.In(key='lr')],
        [sg.T("output_size："), sg.In(key='output_size')],
        [sg.T("num_filters："), sg.In(key='num_filters')],
        [sg.T("kernel_sizes："), sg.In(key='kernel_sizes')],
        [sg.T("dropout："), sg.In(key='dropout')],
    ]
    output = [
        #[sg.Output(size=(30, 20))],
        [sg.B("开始训练", key="开始训练")]
    ]
    layout = [
        [sg.Col(siber), sg.Col(input), sg.Col(output)]
    ]

    return layout
