#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-04-18 19:00:00
@ LastEditors: Yega
@ Description: maintain an array and access it by location through property names
'''


class SliceByAttribute:
    def __init__(self, slice_attr, slice_map, setattr_callback=None) -> None:
        self.__inner_slice_attr__ = slice_attr
        self.__attr_map__ = slice_map
        self.__setattr_callback__ = setattr_callback
        if isinstance(slice_map, tuple):
            self.__attr_map__ = dict()
            for i, k in enumerate(slice_map):
                self.__attr_map__[k] = (i, i+1)

    def __getattr__(self, name):
        _min, _max = self.__attr_map__[name]
        if 1 == _max-_min:
            return self.__inner_slice_attr__[_min]
        return self.__inner_slice_attr__[_min:_max]

    def __setattr__(self, key, value):
        if '__attr_map__' in self.__dict__ and key != '__attr_map__' and key in self.__attr_map__:
            if self.__setattr_callback__:
                self.__setattr_callback__(key, value)
            _min, _max = self.__attr_map__[key]
            if 1 == _max-_min:
                self.__inner_slice_attr__[_min] = value
            else:
                self.__inner_slice_attr__[_min:_max] = value
        else:
            super().__setattr__(key, value)

    @property
    def state(self):
        return self.__inner_slice_attr__

    @state.setter
    def state(self, value):
        if not isinstance(value, type(self.__inner_slice_attr__)):
            raise TypeError(value, type(self.__inner_slice_attr__))
        if len(value) != len(self.__inner_slice_attr__):
            raise IndexError(len(self.__inner_slice_attr__))
        self.__inner_slice_attr__ = value
