#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-07-09 00:00:00
@ LastEditors: Yega
@ Description: Record Simulation log object
'''
import io
import pickle
import time
import datetime
import os
import numpy as np
import threading
import socket
import json

LOGFMT = r"log_%Y-%m-%d-%H-%M-%S"  # 日志文件前缀名格式
LOGSUFFIX = ".pkl"  # 日志文件后缀名
PICKSEPARATOR = b"###log###\n"  # pickle分隔符


def get_lastest_pkl_filename(dirs):
    """根据文件名获取最新的日期文件

    Args:
        dirs (str): 工作区的相对路径，文件目录名称

    Returns:
        str: 文件名称LOGFMT+LOGSUFFIX, Default to None.
    """
    fileDir = os.getcwd()
    fileDir = os.path.join(fileDir, dirs)
    fileList = os.listdir(fileDir)
    file_name_latest: datetime.datetime = None
    for file in fileList:
        if file.endswith('.pkl'):
            t_file_name_latest = datetime.datetime.strptime(file[:-4], LOGFMT)
            if not file_name_latest:
                file_name_latest = t_file_name_latest
            if file_name_latest < t_file_name_latest:
                file_name_latest = t_file_name_latest
    if not file_name_latest:
        return
    return file_name_latest.strftime(LOGFMT)+LOGSUFFIX


class RecordOBJ:
    def __init__(self, *args, **kwargs) -> None:
        """日志记录的对象,包含参数,可选参数,可选参数变为类的成员.
        """
        self.field = None
        self.__inline_kwargs = kwargs
        self.__inline_args = args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_args(self) -> tuple:
        """返回当前类的参数

        Returns:
            tuple: args
        """
        return self.__inline_args

    def get_kwargs(self) -> dict:
        """返回类的可选参数

        Returns:
            dict: kwargs
        """
        return self.__inline_kwargs


class Recorder:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(Recorder, cls).__new__(cls)
        return cls._instance

    def __init__(self, filename=None, playback=False, dirs='record_log', send_json_addr='') -> None:
        """日志记录

        Args:
            filename (str, optional): 指定需要记录的日志名称.默认自动创建指定格式的文件名.
            playback (bool, optional): 指定当前是否为回放日志模式. 'True'为写入记录, Defaults to False.
            dirs (str, optional): 指定当前工作区的写入的文件夹. Defaults to 'record_log'.
        """
        self.send_json = True if ':' in send_json_addr else False
        if self.send_json:
            self.send_json_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # self.send_json_sock.settimeout(0.01)
            tmp = send_json_addr.split(':')
            self.send_json_address = (tmp[0], int(tmp[1]))
            # self.send_json_num = 50
            # self.send_json_buf_list = []
        if playback:  # 回放时无需添加文件名，自动取最新的文件
            self.filename = filename if filename else get_lastest_pkl_filename(dirs)
        else:
            self.filename = filename if filename else self.__file_create_func()
        if not self.filename:
            print('not found file.')
            exit(0)
        mode = 'rb' if playback else 'wb+'
        self.bytesio = io.BytesIO()
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            print(f"Folder created {dirs}")
        self.filename = dirs+os.sep+self.filename
        self.pkfile = open(self.filename, mode)

    def readall(self):
        """读取文件所有日志,通过字段名和变量名索引

        Raises:
            NotImplementedError: 0,1,2维数据将会被展平叠加在一起,高维数据未实现读取,需自行处理。

        Returns:
            TMPOBJ: Object has members & fields.
        """
        class TMPOBJ:
            pass

        tmp_obj = TMPOBJ()
        while True:
            res = self.read()
            if not res:
                break
            if not hasattr(tmp_obj, res.field):
                setattr(tmp_obj, res.field, TMPOBJ())
            field_obj = getattr(tmp_obj, res.field)
            for key, value in res.get_kwargs().items():
                value = np.array(value)
                ndim = value.ndim
                if ndim > 0:
                    value = value.flatten()
                if not hasattr(field_obj, key):
                    setattr(field_obj, key, value)
                    continue
                key_obj = getattr(field_obj, key)
                if ndim == 0:
                    new_key_obj = np.append(key_obj, value)
                elif ndim == 1 or ndim == 2:
                    new_key_obj = np.vstack((key_obj, value))
                else:
                    raise NotImplementedError("too big!", value)
                setattr(field_obj, key, new_key_obj)
        return tmp_obj

    def read(self) -> RecordOBJ:
        """读取文件一行pickle的对象字节流,变换成一个带字段名的对象.

        Returns:
            RecordOBJ: 可索引的对象,None代表没有读到数据.
        """
        buf = []
        while True:
            line = self.pkfile.readline()
            if not line:
                break
            buf.append(line)
            if line.endswith(b'.'+PICKSEPARATOR):  # pickle unsafe! & endline=PICKSEPARATOR
                obj = pickle.loads(b''.join(buf).replace(PICKSEPARATOR, b''))
                return obj
        return

    def write(self, *args, **kwargs):
        """向日志中写写入数据，接受参数和可选参数，推荐使用可选参数
        参数第一个必须为字段名

        Raises:
            ValueError: 未定义字段名，无法写入
        """
        if len(args) <= 0:
            raise ValueError("not define field name!")
        obj = RecordOBJ(*args, **kwargs)
        obj.field = args[0]
        if self.send_json:
            sub_tmp = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    sub_tmp[k] = v.reshape(-1).tolist()
                else:
                    sub_tmp[k] = v
            tmp = {args[0]: sub_tmp}
            print(tmp)
            print(json.dumps(tmp))
            data = json.dumps(tmp).encode('utf-8')
            self.send_json_sock.sendto(data, self.send_json_address)
            # self.send_json_buf_list.append(data)
            # if len(self.send_json_buf_list) >= self.send_json_num:
            #     for buf in self.send_json_buf_list:
            #         self.send_json_sock.sendto(buf, self.send_json_address)
            #     self.send_json_buf_list = []
        obj_bytes = pickle.dumps(obj)
        self.bytesio.write(obj_bytes)
        self.bytesio.write(PICKSEPARATOR)
        if self.bytesio.tell() > 1024*1024:
            self.__write_file()

    def flush(self):
        """立即将缓冲区内容写入文件
        """
        self.__write_file()

    def __write_file(self):
        data = self.bytesio.getvalue()
        self.bytesio.seek(0, 0)
        self.bytesio.truncate()
        self.pkfile.seek(0, 2)
        self.pkfile.write(data)
        self.pkfile.flush()

    def __file_create_func(self):
        """产生一个当前时间的文件名

        Returns:
            str: 预定义格式的文件名
        """
        loca = time.strftime(LOGFMT)
        filename = str(loca)+LOGSUFFIX
        return filename

    def close(self):
        if self.bytesio.closed:
            return
        if self.pkfile.closed:
            return
        if self.bytesio.tell() > 0:
            self.__write_file()
        self.bytesio.close()
        self.pkfile.close()

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass


if __name__ == "__main__":
    import numpy as np
    channels = np.array([[1, 0, -1000,],
                        [50, 0, 2,],
                        [1, 5, 0,]])
    state = np.array([0, 0, -1000,
                      50, 0, 0,
                      1, 0, 0, 0,
                      0, 0, 0,
                      0, 0, 0], dtype=float)
    record = Recorder(send_json_addr='127.0.0.1:9870')
    record.write("FaM", deltaT=1, channels=channels, state=np.arange(5),
                 M_tb=np.ones(5), M_ab=np.ones(5), M_Lb=np.ones(5),)
    record.write("FaM", deltaT=2, channels=channels, state=np.arange(5),
                 M_tb=np.ones(5), M_ab=np.ones(5), M_Lb=np.ones(5),)
    record.write("FaM", np.arange(5), np.ones(5), deltaT=3, channels=channels, state=np.arange(5),
                 M_tb=np.zeros(5), M_ab=np.zeros(5), M_Lb=np.zeros(5),)
    record.close()
    # record = Recorder(playback=True)
    # res = record.read()
    # print(res.field, res.state)
    # res = record.read()
    # print(res.state)
    # res = record.read()
    # print(res.state, res.get_args(), res.get_kwargs())
    # record.close()

    # record = Recorder(playback=True)
    # obj = record.readall()
    # print()
    # print("channels ", obj.FaM.channels)
    # print("M_tb ", obj.FaM.M_tb)
    # print("deltaT ", obj.FaM.deltaT)
    # record.close()

# if __name__ == "__main__":
#     import numpy as np
#     from tensorboardX import SummaryWriter
#     writer = SummaryWriter(log_dir='logs')
#     idx = -1
#     record = Recorder(playback=True)
#     while True:
#         obj = record.read()
#         if obj is None:
#             break
#         # for i in obj.state:
#         idx += 1
#         writer.add_scalar('scalar/test', obj.state, idx)
#         # writer.add_scalars('scalar/scalars_test', {'xsinx': epoch *
#         #                                            np.sin(epoch), 'xcosx': epoch * np.cos(epoch)}, epoch)
#     writer.close()
#     print(idx)
