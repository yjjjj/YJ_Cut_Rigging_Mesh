# coding:utf-8

"""
-------------------------------------------------------------------------------

@author: LiuJingJing
@email : iex@live.com

--------------------------------------------------------------------------------
"""

import os
from .pyside import *
from .image_button_box import ImageButtonBox


PATH = os.path.dirname(os.path.abspath(__file__))


class UIMain(QDialog):
    def __init__(self, parent=None):
        super(UIMain, self).__init__(parent)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.folder_path = None
        self.setWindowTitle("YJ Cut Rigging Mesh 1.0.3")
        self.resize(390, 500)
        self.setup_ui()

    def setup_ui(self):
        # 创建菜单栏
        self.menu_bar = QMenuBar()
        file_menu = self.menu_bar.addMenu("Edit")
        
        # 添加菜单项
        self.menu_parent = QAction("Parent", self)
        self.menu_copy_skin = QAction("Copy Skin", self)
        self.menu_reduce_skin_mesh = QAction("Reduce Mesh", self)
        self.menu_mirror_select = QAction("Mirror Select X", self)
        file_menu.addAction(self.menu_parent)
        file_menu.addAction(self.menu_copy_skin)
        file_menu.addAction(self.menu_reduce_skin_mesh)
        file_menu.addAction(self.menu_mirror_select)

        self.add_attr_btn = QPushButton()
        self.add_attr_btn.setIcon(QIcon(os.path.join(self.current_path, 'icons', 'add_attr.png')))
        self.add_attr_btn.setIconSize(QSize(23, 23))
        self.add_attr_btn.setFlat(True)

        # Show Ctrl Layout
        self.ctrl_text = QLabel("Main Ctrl:")
        self.ctrl_text.setFixedWidth(70)
        self.ctrl_line = QLineEdit()
        self.ctrl_line.setFocusPolicy(Qt.ClickFocus)
        self.ctrl_line.clearFocus()
        self.ctrl_line.setFixedHeight(25)
        self.load_ctrl_btn = QPushButton("···")
        self.load_ctrl_btn.setFixedWidth(35)

        load_ctrl_layout = QHBoxLayout()
        load_ctrl_layout.addWidget(self.ctrl_text)
        load_ctrl_layout.addWidget(self.ctrl_line)
        load_ctrl_layout.addWidget(self.load_ctrl_btn)

        # Skin Mesh Layout
        self.mesh_text = QLabel("Skin Mesh:")
        self.mesh_text.setFixedWidth(70)
        self.mesh_line = QLineEdit()
        self.mesh_line.setFocusPolicy(Qt.ClickFocus)
        self.mesh_line.clearFocus()
        self.mesh_line.setFixedHeight(25)
        self.load_mesh_btn = QPushButton("···")
        self.load_mesh_btn.setFixedWidth(35)

        load_mesh_layout = QHBoxLayout()
        load_mesh_layout.addWidget(self.mesh_text)
        load_mesh_layout.addWidget(self.mesh_line)
        load_mesh_layout.addWidget(self.load_mesh_btn)

        self.disc_radio_btn = QRadioButton("disc")
        self.disc_radio_btn.setChecked(1)
        self.select_radio_btn = QRadioButton("select")
        self.create_select_btn = ImageButtonBox(os.path.join(PATH, "icons", "create_select.png"), (25,25))
        tool_bar_layout = QHBoxLayout()
        tool_bar_layout.addWidget(self.disc_radio_btn)
        tool_bar_layout.addWidget(self.select_radio_btn)
        tool_bar_layout.addStretch(1)
        tool_bar_layout.addWidget(self.create_select_btn)

        self.template_joint_clear_btn = ImageButtonBox(os.path.join(PATH, "icons", "clear.png"), (23,23))
        self.joint_list_add_btn = ImageButtonBox(os.path.join(PATH, "icons", "add.png"), (23,23))
        self.joint_list_widget = JointListWidget()
        self.joint_list_widget.rootJnt = None
        self.joint_list_widget.setFocusPolicy(Qt.ClickFocus)
        self.joint_list_widget.clearFocus()
        
        joint_btn_layout = QHBoxLayout()
        joint_btn_layout.addStretch(1)
        joint_btn_layout.addWidget(self.template_joint_clear_btn)
        joint_btn_layout.addWidget(self.joint_list_add_btn)
        joint_list_layout = QVBoxLayout()
        joint_list_layout.addLayout(joint_btn_layout)
        joint_list_layout.addWidget(self.joint_list_widget)

        self.mesh_list_add_btn = ImageButtonBox(os.path.join(PATH, "icons", "add.png"), (23,23))
        self.mesh_list_clear_btn = ImageButtonBox(os.path.join(PATH, "icons", "clear.png"), (23,23))
        self.mesh_list_widget = MeshListWidget()
        self.mesh_list_widget.setFocusPolicy(Qt.ClickFocus)
        self.mesh_list_widget.clearFocus()
        mesh_btn_layout = QHBoxLayout()
        mesh_btn_layout.addStretch(1)
        mesh_btn_layout.addWidget(self.mesh_list_clear_btn)
        mesh_btn_layout.addWidget(self.mesh_list_add_btn)
        mesh_list_layout = QVBoxLayout()
        mesh_list_layout.addLayout(mesh_btn_layout)
        mesh_list_layout.addWidget(self.mesh_list_widget)

        main_list_widget = QHBoxLayout()
        main_list_widget.addLayout(joint_list_layout)
        main_list_widget.addLayout(mesh_list_layout)

        self.create_btn = QPushButton("CREATE")
        self.create_btn.setFixedHeight(36)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setFrameShadow(QFrame.Sunken)

        layout = QVBoxLayout(self)
        layout.addWidget(line)
        layout.addLayout(load_ctrl_layout)
        layout.addLayout(load_mesh_layout)
        layout.addSpacing(5)
        layout.addWidget(line2)
        layout.addLayout(tool_bar_layout)
        layout.addWidget(line3)

        layout.addLayout(main_list_widget)
        layout.setMenuBar(self.menu_bar)
        # layout.addStretch(1)
        layout.addWidget(self.create_btn)


class JointListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.set_root_action = QAction("Set Root", self)
        self.delete_action = QAction("Delete", self)

    def showContextMenu(self, pos):
        item = self.itemAt(pos)
        if item:
            menu = QMenu()
            menu.addAction(self.set_root_action)
            menu.addAction(self.delete_action)

            menu.exec_(self.mapToGlobal(pos))


class MeshListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.delete_action = QAction("Delete", self)

    def showContextMenu(self, pos):
        item = self.itemAt(pos)
        if item:
            menu = QMenu()
            menu.addAction(self.delete_action)

            menu.exec_(self.mapToGlobal(pos))


class MyButton(QPushButton):
    def __init__(self):
        super(MyButton, self).__init__()
        self.initUI()

    def initUI(self):
        # 创建一个右键菜单
        self.context_menu = QMenu(self)
        self.action = QAction("Update", self)
        self.context_menu.addAction(self.action)

        # 将右键菜单关联到按钮
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.show()

    def showContextMenu(self, pos):
        self.context_menu.exec_(self.mapToGlobal(pos))