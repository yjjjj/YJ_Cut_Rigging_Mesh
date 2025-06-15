# coding:utf-8

"""
-------------------------------------------------------------------------------

@author: LiuJingJing
@email : iex@live.com

--------------------------------------------------------------------------------
"""

import os
import traceback
import maya.cmds as cmds
import pymel.core as pm

from .core import utils
from .gui import ui_main
from .gui.ui_connector import connector
from .gui.pyside import *

from imp import reload
reload(utils)
reload(ui_main)


PATH = os.path.dirname(os.path.abspath(__file__))



class CutMesh(ui_main.UIMain):
    def __init__(self, parent=None):
        super(CutMesh, self).__init__(parent)
        cmds.lockNode('initialShadingGroup', lock=0, lockUnpublished=0)

        self.load_ctrl_btn.clicked.connect(self.load_ctrl_btn_cmd)
        self.load_mesh_btn.clicked.connect(self.load_mesh_btn_cmd)
        self.joint_list_add_btn.clicked.connect(self.joint_list_add_btn_cmd)
        self.template_joint_clear_btn.clicked.connect(self.template_joint_clear_btn_cmd)
        self.mesh_list_add_btn.clicked.connect(self.mesh_list_add_btn_cmd)
        self.mesh_list_clear_btn.clicked.connect(self.mesh_list_clear_btn_cmd)
        self.create_select_btn.clicked.connect(self.start_create_select_mesh_btn_cmd)
        self.create_btn.clicked.connect(self.start_create_cut)

        self.menu_parent.triggered.connect(self.menu_parent_cmd)
        self.menu_copy_skin.triggered.connect(self.menu_copy_skin_cmd)
        self.menu_reduce_skin_mesh.triggered.connect(self.start_menu_reduce_skin_mesh_cmd)
        self.menu_mirror_select.triggered.connect(self.start_menu_mirror_select_cmd)

        self.joint_list_widget.set_root_action.triggered.connect(self.set_root_action_cmd)
        self.joint_list_widget.delete_action.triggered.connect(self.delete_joint_action_cmd)
        self.mesh_list_widget.delete_action.triggered.connect(self.delete_mesh_action_cmd)
        self.joint_list_widget.itemClicked.connect(self.joint_list_on_clicked)
        self.mesh_list_widget.itemClicked.connect(self.mesh_list_on_clicked)

        self.update_list_attr()

    def start_menu_mirror_select_cmd(self):
        cmds.undoInfo(openChunk=True)
        self.menu_mirror_select_cmd()
        cmds.undoInfo(closeChunk=True)

    def start_menu_reduce_skin_mesh_cmd(self):
        cmds.undoInfo(openChunk=True)
        self.menu_reduce_skin_mesh_cmd()
        cmds.undoInfo(closeChunk=True)

    def start_create_select_mesh_btn_cmd(self):
        cmds.undoInfo(openChunk=True)
        self.create_select_mesh_btn_cmd()
        cmds.undoInfo(closeChunk=True)

    def start_create_cut(self):
        cmds.undoInfo(openChunk=True)
        self.create_cut()
        cmds.undoInfo(closeChunk=True)

    def get_joint_list_widget(self):
        jnt_items = []
        for x in range(self.joint_list_widget.count()):
            jnt_items.append(self.joint_list_widget.item(x).text())

        return jnt_items

    def get_mesh_list_widget(self):
        mesh_items = []
        for x in range(self.mesh_list_widget.count()):
            mesh_items.append(self.mesh_list_widget.item(x).text())

        return mesh_items

    def load_ctrl_btn_cmd(self):
        sel = cmds.ls(sl=True)
        if sel:
            self.ctrl_line.setText(sel[0])

    def load_mesh_btn_cmd(self):
        sel = cmds.ls(sl=True)
        if sel:
            self.mesh_line.setText(sel[0])

    # 菜单(创建约束)
    def menu_parent_cmd(self):
        sel = cmds.ls(sl=True)

        if sel:
            root_obj = sel[0]
            child_obj = sel[1]

            hilite = cmds.listHistory(child_obj, pdo=True)
            skin_cluster = cmds.ls(hilite, type="skinCluster")
            if skin_cluster:
                cmds.skinCluster(child_obj, e=True, ub=True)

            parent_constraint = cmds.listRelatives(child_obj, allDescendents=True, type="parentConstraint")
            if parent_constraint:
                cmds.delete(parent_constraint)

            utils.set_pivot_parent(root_obj, child_obj)
            cmds.setAttr("{}.inheritsTransform".format(child_obj), 1)


    # 菜单(拷贝权重)
    def menu_copy_skin_cmd(self):
        skin_mesh = self.mesh_line.text()
        if not skin_mesh:
            self.message(u"Please load the skin model first \n( 请先加载蒙皮模型 )")
            return

        sel = cmds.ls(sl=True)
        for i in sel:
            parent_constraint = cmds.listRelatives(i, allDescendents=True, type="parentConstraint")
            if parent_constraint:
                cmds.delete(parent_constraint)

            utils.copy_skin(skin_mesh, i)
            cmds.setAttr("{}.inheritsTransform".format(i), 0)
    

    # 菜单(skin模型减面)
    def menu_reduce_skin_mesh_cmd(self):
        try:
            sel = cmds.ls(sl=True, type="transform")

            ctrl_name = self.ctrl_line.text()
            if not ctrl_name:
                self.message(u"Please load the ctrl first \n( 请先加载控制器 )")
                return

            if not cmds.objExists("REDUCEMESH_GRP"):
                reduce_grp = cmds.group(em=True, n="REDUCEMESH_GRP")
            else:
                reduce_grp = "REDUCEMESH_GRP"

            if cmds.objExists("MASTER_CUTMESH_GRP"):
                cmds.parent(reduce_grp, "MASTER_CUTMESH_GRP")

            for mesh in sel:
                mesh_skin = mesh
                default_hilite = cmds.listHistory(mesh, pdo=True)
                default_skin_cluster = cmds.ls(default_hilite, type="skinCluster")
                default_skin_num = len(cmds.skinCluster(default_skin_cluster, query=True, inf=True))

                hilite = cmds.listHistory(mesh, pdo=True)
                blend_shapes = cmds.ls(hilite, type="blendShape")

                if blend_shapes:
                    for bs in blend_shapes:
                        input_mesh = cmds.connectionInfo("{}.inputTarget[0].inputTargetGroup[0]"
                                                         ".inputTargetItem[6000].inputGeomTarget".format(bs),
                                                         sourceFromDestination=True)
                        if input_mesh:
                            input_mesh = input_mesh.split('.worldMesh')[0]
                            input_hilite = cmds.listHistory(input_mesh, pdo=True)
                            input_skin_cluster = cmds.ls(input_hilite, type="skinCluster")
                            if input_skin_cluster:
                                input_skin_num = len(cmds.skinCluster(input_skin_cluster, query=True, inf=True))
                                if input_skin_num > default_skin_num:
                                    mesh_skin = input_mesh

                reduce_mesh = cmds.duplicate(mesh, n="{}_REDUCEMESH".format(mesh))[0]
                cmds.parent(reduce_mesh, w=True)
                cmds.polyReduce(reduce_mesh, ver=1, trm=0, shp=0, keepBorder=1, keepMapBorder=1,
                                keepColorBorder=1, keepFaceGroupBorder=1, keepHardEdge=1, keepCreaseEdge=1,
                                keepBorderWeight=0.5, keepMapBorderWeight=0.5, keepColorBorderWeight=0.5,
                                keepFaceGroupBorderWeight=0.5, keepHardEdgeWeight=0.5, keepCreaseEdgeWeight=0.5,
                                useVirtualSymmetry=0, symmetryTolerance=0.01, sx=0, sy=1, sz=0, sw=0,
                                preserveTopology=1, keepQuadsWeight=1, vertexMapName="", cachingReduce=1,
                                ch=0, p=70, vct=0, tct=0, replaceOriginal=1)

                utils.copy_skin(mesh_skin, reduce_mesh)
                self.set_low_high_switch(ctrl_name, mesh, reduce_mesh)

                cmds.parent(reduce_mesh, reduce_grp)
                
        except Exception as e:
            self.message("Error: {} \n ( 出现错误，请Ctrl+Z退回 )".format(e))
            traceback.print_exc()


    # 菜单(创建模型镜像)
    def menu_mirror_select_cmd(self):
        sel_list = cmds.ls(sl=True, type="transform")

        if sel_list:
            master_grp, cut_grp = self.get_master_grp()
            utils.mirror_mesh(sel_list, cut_grp)

        jnt_items = self.get_joint_list_widget()
        for mesh in sel_list:
            mirror_name = mesh.replace('_L', '_R').replace('_l', '_r').replace('_CUTMESH', '').replace('_CUS', '')
            if '_R' in mesh or '_r' in mesh:
                mirror_name = mesh.replace('_R', '_L').replace('_r', '_l').replace('_CUTMESH', '').replace('_CUS', '')

            # 添加到ListWidget列表
            if mirror_name not in jnt_items:
                self.joint_list_widget.insertItem(0, mirror_name)

                # 添加骨骼属性连接到Master组
                self.connect_joint_attr(pm.PyNode(mirror_name), pm.PyNode(master_grp))

    # ListWidget右键菜单设置root骨骼
    def set_root_action_cmd(self):
        if not self.joint_list_widget.rootJnt:
            item = self.joint_list_widget.currentItem()

            if cmds.objExists(item.text()):
                item.setBackground(QColor('#126845'))

                self.joint_list_widget.rootJnt = item.text()
                self.joint_list_widget.clearSelection()

    # ListWidget右键菜单删除Item
    def delete_joint_action_cmd(self):
        item = self.joint_list_widget.currentItem()
        jnt = pm.PyNode(item.text())
        jnt.message.disconnect()

        row = self.joint_list_widget.currentRow()
        self.joint_list_widget.takeItem(row)

    def delete_mesh_action_cmd(self):
        item = self.mesh_list_widget.currentItem()
        mesh = pm.PyNode(item.text())
        mesh.message.disconnect()

        row = self.mesh_list_widget.currentRow()
        self.mesh_list_widget.takeItem(row)

    def joint_list_on_clicked(self, *args):
        item = self.joint_list_widget.currentItem()
        jnt_name = item.text()
        if cmds.objExists(jnt_name):
            cmds.select(jnt_name)

    def mesh_list_on_clicked(self, *args):
        item = self.mesh_list_widget.currentItem()
        mesh_name = item.text()
        if cmds.objExists(mesh_name):
            cmds.select(mesh_name)

    # 添加骨骼到列表
    def joint_list_add_btn_cmd(self):
        if cmds.objExists("MASTER_CUTMESH_GRP"):
            jnt_items = self.get_joint_list_widget()

            sel = cmds.ls(sl=True)
            for i in sel:
                if i not in jnt_items:
                    self.joint_list_widget.insertItem(0, i)
                    
                    self.connect_joint_attr(pm.PyNode(i), pm.PyNode("MASTER_CUTMESH_GRP"))

    # 添加切割模型片到列表
    def mesh_list_add_btn_cmd(self):
        if cmds.objExists("CUTMESH_GRP"):
            mesh_items = self.get_mesh_list_widget()

            sel = pm.selected()
            for obj in sel:
                if isinstance(obj, pm.nt.Transform):
                    shapes = obj.getShapes()
                    if shapes and isinstance(shapes[0], pm.nt.Mesh):
                        if obj.name() not in mesh_items:
                            self.mesh_list_widget.insertItem(0, obj.name())
                            self.connect_mesh_attr(obj, pm.PyNode("CUTMESH_GRP"))

                    else:
                        all_descendants = obj.listRelatives(allDescendents=True)
                        models = [obj for obj in all_descendants if isinstance(obj, pm.nt.Mesh)]
                        model_transforms = [model.getParent() for model in models]
                        for mesh in model_transforms:
                            if mesh.name() not in mesh_items:
                                self.mesh_list_widget.addItem(mesh.name())
                                self.connect_mesh_attr(mesh, pm.PyNode("CUTMESH_GRP"))


    def template_joint_clear_btn_cmd(self):
        self.joint_list_widget.clear()
        self.joint_list_widget.rootJnt = None
        pm.PyNode("MASTER_CUTMESH_GRP").list_joint_attr.disconnect()

    def mesh_list_clear_btn_cmd(self):
        self.mesh_list_widget.clear()
        pm.PyNode("CUTMESH_GRP").list_mesh_attr.disconnect()

    # 创建切割片模型
    def create_cut_mesh(self, mesh_name):
        if self.disc_radio_btn.isChecked():
            cut_mesh = utils.create_poly_disc(mesh_name)
        if self.select_radio_btn.isChecked():
            sel = cmds.ls(sl=True, fl=True)
            sel_jnt = sel[-1]
            sel_edges = sel[:-1]
            cmds.select(sel_edges)
            cv = utils.edges_to_curve("{}_CUS".format(mesh_name))
            cut_mesh = utils.curve_loft_mesh(cv, thickness=0.004)
            cmds.polyColorPerVertex(cut_mesh, rgb=(0.7, 0.1, 0.5), alpha=0.1, colorDisplayOption=True)
            shape = cmds.ls(cut_mesh, dag=True, s=True, head=True)[0]
            cmds.setAttr(f"{shape}.overrideEnabled", 1)
            cmds.setAttr(f"{shape}.overrideColor", 21)
            cmds.select(cut_mesh)
            cmds.CenterPivot(cut_mesh)
            cmds.delete(cv)

        return cut_mesh

    # 根据选择骨骼创建切割模型
    def create_select_mesh_btn_cmd(self):
        try:
            skin_mesh = self.mesh_line.text()
            if not skin_mesh:
                self.message(u"Please load the skin model first \n( 请先加载蒙皮模型 )")
                return

            joints = cmds.ls(sl=True, type="joint")

            if joints:
                jnt_items = self.get_joint_list_widget()

                for jnt in joints:
                    if self.disc_radio_btn.isChecked():
                        cut_mesh = self.create_cut_mesh("{}_CUTMESH".format(jnt))
                        self.create_constrain(jnt, cut_mesh)
                        self.set_scale_cut_mesh(jnt, skin_mesh, cut_mesh)

                    if self.select_radio_btn.isChecked():
                        cut_mesh = self.create_cut_mesh(jnt)
                    
                    master_grp, cut_grp = self.get_master_grp()
                    cmds.parent(cut_mesh, cut_grp)
                    cmds.makeIdentity(cut_mesh, apply=True, t=1, r=1, s=1, n=0, pn=1)

                    # 添加到ListWidget列表
                    if jnt not in jnt_items:
                        self.joint_list_widget.insertItem(0, jnt)

                    # 添加骨骼属性连接到Master组
                    self.connect_joint_attr(pm.PyNode(jnt), pm.PyNode(master_grp))

        except Exception as e:
            self.message(u"Error: {} \n ( 出现错误，请Ctrl+Z退回 )".format(e))
            traceback.print_exc()

    def create_constrain(self, jnt, cut_mesh):
        point_offset = (0, 0, 0)
        orient_offset = (0, 0, 0)

        axis = utils.get_joint_aim_axis(jnt)
        print("axis:", axis)
        if axis == "X":
            orient_offset = (0, 0, 90)
        if axis == "Y":
            orient_offset = (0, 90, 0)
        if axis == "Z":
            orient_offset = (90, 0, 0)

        point_constrain = cmds.pointConstraint(jnt, cut_mesh, offset=point_offset, weight=1)
        orient_constraint = cmds.orientConstraint(jnt, cut_mesh, offset=orient_offset, weight=1)
        cmds.delete(point_constrain, orient_constraint)

    # 创建切割
    def create_cut(self):
        try:
            ctrl_name = self.ctrl_line.text()
            skin_mesh = self.mesh_line.text()
            if not ctrl_name:
                self.message(u"Please load the ctrl first \n( 请先加载控制器 )")
                return

            if not skin_mesh:
                self.message(u"Please load the skin model first \n( 请先加载蒙皮模型 )")
                return

            if not self.joint_list_widget.rootJnt:
                self.message(u"Please set up the root skeleton first \n 请先设置 root 骨骼")
                return

            cut_mesh_list = self.get_mesh_list_widget()
            new_mesh_body = cmds.duplicate(skin_mesh, name="{}_TEMP_BODY".format(skin_mesh))[0]
            new_cut_mesh_list = cmds.duplicate(cut_mesh_list)
            cmds.parent(new_mesh_body, w=True)
            main_cut_mesh = cmds.polyCBoolOp(new_mesh_body,
                                             new_cut_mesh_list,
                                             op=2,
                                             ch=0,
                                             preserveColor=0,
                                             classification=2,
                                             name='{}_SPLITMESH_GRP'.format(skin_mesh))[0]
            # print(main_cut_mesh)

            # 分离模型
            separate_mesh = cmds.polySeparate(main_cut_mesh, ch=0, name="{}_SPLITMESH#".format(skin_mesh))
            # print("separate_mesh:",separate_mesh)

            # 给分离模型添加材质
            material = utils.get_material(skin_mesh)
            if material:
                shading_engine = material["shading_engine"][0]
                cmds.sets(separate_mesh, e=1, forceElement=shading_engine) 

            # 创建约束-----------------------------------------------
            # 得到所有模型中心点
            all_mesh_center_point = dict()
            for mesh in separate_mesh:
                mesh_center_point = utils.get_mesh_center_point(mesh)
                all_mesh_center_point[mesh_center_point] = mesh

                # 清除点颜色属性
                vtx_color = utils.check_vertex_colors(mesh)
                if vtx_color:
                    cmds.polyColorPerVertex(mesh, remove=True)

            # 得到所有骨骼中心点
            jnt_items = self.get_joint_list_widget()

            all_jnt_center_point = dict()
            for jnt in jnt_items:
                # print(jnt)
                if cmds.objExists(jnt):
                    jnt_center_point = utils.get_jnt_middle_point(jnt)
                    # loc = cmds.spaceLocator(p=jnt_center_point)
                    # print(jnt_center_point)

                    # 找到最接近骨骼中心的模型
                    closest_point = utils.find_closest_point(jnt_center_point, all_mesh_center_point.keys())
                    closest_mesh = all_mesh_center_point.get(closest_point)
                    parent_constraint = cmds.listRelatives(closest_mesh, allDescendents=True, type="parentConstraint")
                    
                    # 创建约束
                    if not parent_constraint:
                        utils.set_pivot_parent(jnt, closest_mesh)

                    all_jnt_center_point[(jnt_center_point[0], jnt_center_point[1], jnt_center_point[2])] = jnt
            # ------------------------------------------------------

            # 修复没有被约束的模型
            for point, mesh in all_mesh_center_point.items():
                # print(mesh, point)
                parent_constraint = cmds.listRelatives(mesh, allDescendents=True, type="parentConstraint")
                if not parent_constraint:
                    closest_point = utils.find_closest_point(point, all_jnt_center_point.keys())
                    closest_jnt = all_jnt_center_point.get(closest_point)
                    utils.set_pivot_parent(closest_jnt, mesh)

            master_grp, cut_grp = self.get_master_grp()
            cmds.parent(main_cut_mesh, master_grp)
            # cmds.setAttr("{}.visibility".format(skin_mesh), 0)
            cmds.setAttr("{}.visibility".format(cut_grp), 0)
            cmds.delete(new_mesh_body)
            cmds.select(cl=True)

            # root根骨骼缩放约束切割模型组
            cmds.scaleConstraint(self.joint_list_widget.rootJnt, main_cut_mesh, weight=1)

            # 添加高低模切换显示
            self.set_low_high_switch(ctrl_name, skin_mesh, main_cut_mesh)

        except Exception as e:
            self.message(u"Error: {} \n ( 出现错误，请Ctrl+Z退回 )".format(e))
            traceback.print_exc()

    # 添加高低模切换显示
    def set_low_high_switch(self, ctrl_name, skin_mesh, low_mesh):
        ctrl_condition = "Cut_Mesh_Show_Switch"
        attr_name, low_index = utils.get_low_attr_index(ctrl_name)

        if not cmds.objExists(ctrl_condition):
            ctrl_condition = cmds.createNode("condition", n="Cut_Mesh_Show_Switch")
            cmds.setAttr("{}.secondTerm".format(ctrl_condition), low_index)
            cmds.setAttr("{}.colorIfTrueR".format(ctrl_condition), 1)
            cmds.setAttr("{}.colorIfFalseR".format(ctrl_condition), 0)
            cmds.setAttr("{}.colorIfTrueG".format(ctrl_condition), 0)
            cmds.setAttr("{}.colorIfFalseG".format(ctrl_condition), 1)
            cmds.connectAttr("{}.{}".format(ctrl_name, attr_name), ".firstTerm".format(ctrl_condition))

        main_cut_mesh_node = pm.PyNode(low_mesh)
        main_cut_mesh_node.v.disconnect()
        cmds.connectAttr("{}.outColorR".format(ctrl_condition), "{}.v".format(low_mesh))

        skin_mesh_node = pm.PyNode(skin_mesh)
        skin_mesh_node.v.disconnect()

        cmds.connectAttr("{}.outColorG".format(ctrl_condition), "{}.v".format(skin_mesh))
        cmds.setAttr("{}.{}".format(ctrl_name, attr_name), low_index)

    # 设置切割片的缩放
    def set_scale_cut_mesh(self, jnt, skin_mesh, cut_mesh):
        try:
            distance = utils.cast_ray_from_joint_yz(jnt, skin_mesh)
        except:
            distance = 1
        cluster = cmds.cluster(cut_mesh, name='{}_CS'.format(jnt))[1]
        scale_attr = (distance*1.318, distance*1.318, distance*1.318)
        cmds.xform(cluster, s=scale_attr)
        cmds.delete(cut_mesh, constructionHistory=True)

    def get_master_grp(self):
        if not cmds.objExists("MASTER_CUTMESH_GRP"):
            master_grp = cmds.group(em=True, n="MASTER_CUTMESH_GRP")
            cmds.addAttr(master_grp, longName='list_joint_attr', attributeType='message', multi=True)
        else:
            master_grp = "MASTER_CUTMESH_GRP"

        if not cmds.objExists("CUTMESH_GRP"):
            cut_grp = cmds.group(em=True, n="CUTMESH_GRP")
            cmds.addAttr(cut_grp, longName='list_mesh_attr', attributeType='message', multi=True)
            cmds.parent(cut_grp, master_grp)
        else:
            cut_grp = "CUTMESH_GRP"

        cmds.select(cl=True)
        return master_grp, cut_grp

    def connect_joint_attr(self, obj, master_grp):
        list_attr = master_grp.list_joint_attr.get()
        obj.message.connect(master_grp.list_joint_attr[len(list_attr)])

    def connect_mesh_attr(self, obj, master_grp):
        list_attr = master_grp.list_mesh_attr.get()
        obj.message.connect(master_grp.list_mesh_attr[len(list_attr)])

    def update_list_attr(self):
        if pm.objExists("MASTER_CUTMESH_GRP"):
            master_grp = pm.PyNode("MASTER_CUTMESH_GRP")
            jnt_list_attr = master_grp.list_joint_attr.get()

            for jnt in jnt_list_attr:
                self.joint_list_widget.addItem(jnt.name())

        if pm.objExists("CUTMESH_GRP"):
            cut_grp = pm.PyNode("CUTMESH_GRP")
            mesh_list_attr = cut_grp.list_mesh_attr.get()

            for mesh in mesh_list_attr:
                self.mesh_list_widget.addItem(mesh.name())

    def message(self, message_text):
        msg = u"<span style='color:#FF9933;'> {} </span>".format(message_text)
        cmds.inViewMessage(font="Arial", dragKill=True, fade=True, fontSize=15, position="midCenter", amg=msg)



def run():
    win = CutMesh(connector)
    win.show()

