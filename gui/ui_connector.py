# coding:utf-8


from PySide2 import QtWidgets
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui
from pymel import versions


if versions.current() < 20220000:
    connector = wrapInstance(long(omui.MQtUtil.mainWindow()), QtWidgets.QDialog)
else:
    connector = wrapInstance(int(omui.MQtUtil.mainWindow()), QtWidgets.QDialog)
