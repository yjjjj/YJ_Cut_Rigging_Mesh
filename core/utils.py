# coding:utf-8

"""
-------------------------------------------------------------------------------


@author: LiuJingJing
@email : iex@live.com

--------------------------------------------------------------------------------
"""

import math
import maya.cmds as cmds
import pymel.core as pm
import maya.api.OpenMaya as om


# 得到模型点位置
def get_positions(mesh_name):
    sel = om.MGlobal.getSelectionListByName(mesh_name)
    mobj = sel.getDagPath(0)
    mfn_mesh = om.MFnMesh(mobj)
    vert_array = mfn_mesh.getPoints()
    return vert_array


# 得到模型点位置
def get_vertex_positions(mesh_name, precision=7):
    selection_list = om.MSelectionList()
    selection_list.add(mesh_name)

    dag_path = selection_list.getDagPath(0)

    mesh_fn = om.MFnMesh(dag_path)

    vertex_positions = {(float(str(mesh_fn.getPoint(i).x)[:precision+2]), 
                         float(str(mesh_fn.getPoint(i).y)[:precision+2]), 
                         float(str(mesh_fn.getPoint(i).z)[:precision+2])):
                        "{}.vtx[{}]".format(mesh_name, i)
                        for i in range(mesh_fn.numVertices)}

    return vertex_positions


# 拷贝权重
def copy_skin(source_model, target_model):
    # 获取源模型的SkinCluster
    source_skinCluster = cmds.ls(cmds.listHistory(source_model), type='skinCluster')
    if not source_skinCluster:
        raise RuntimeError(u"源模型没有SkinCluster")
    
    source_skinCluster = source_skinCluster[0]
    
    # 获取源模型SkinCluster的影响骨骼
    joints = cmds.skinCluster(source_skinCluster, query=True, influence=True)
    
    # 为目标模型创建SkinCluster，如果它还没有
    target_skinCluster = cmds.ls(cmds.listHistory(target_model), type='skinCluster')
    if not target_skinCluster:
        target_skinCluster = cmds.skinCluster(joints, target_model, toSelectedBones=True)
        target_skinCluster = target_skinCluster[0]
    else:
        target_skinCluster = target_skinCluster[0]
    
    # 复制权重
    cmds.copySkinWeights(ss=source_skinCluster, ds=target_skinCluster, noMirror=True, surfaceAssociation='closestPoint', influenceAssociation=['name', 'closestJoint'])


# 得到骨骼权重的模型点信息
def get_joint_weights(mesh_name, skin_cluster, joint_name, ignore_below=0.001):
    
    vertices = cmds.ls("{}.vtx[*]".format(mesh_name), fl=True)

    weights = dict()
    for vertex in vertices:
        weight = cmds.skinPercent(skin_cluster, vertex, transform=joint_name, query=True)
        if weight > ignore_below:
            weights[vertex] = weight
    
    return(weights)


# 得到两个骨骼skin重叠的部分
def get_overlap_skin(mesh_name, skin_cluster, joint_group, ignore_below=0.2):
    """
    mesh_name:     蒙皮模型名称
    skin_cluster:  蒙皮cluster名称
    joint_list:    要检测的两个骨骼
    ignore_below： 权重过滤值(默认是0.2以下的权重值忽略)
    """
    weights1 = get_joint_weights(mesh_name, skin_cluster, joint_group[0], ignore_below=ignore_below)
    weights2 = get_joint_weights(mesh_name, skin_cluster, joint_group[1], ignore_below=ignore_below)

    overlap_skin_vtx = set(weights1.keys()) & set(weights2.keys())

    return overlap_skin_vtx


# 选择的模型边转换为曲线
def edges_to_curve(curve_name, curve_spans=None, curve_color=6, curve_width=3):
    """
    curve_spans: 曲线细分数
    curve_color: 曲线颜色
    curve_width: 曲线显示宽度
    """
    edge_curve = cmds.polyToCurve(degree=3, form=2, usm=1, ch=0, n=curve_name)[0]
    cmds.setAttr("{}Shape.overrideEnabled".format(edge_curve), 1)
    cmds.setAttr("{}Shape.overrideColor".format(edge_curve), curve_color)
    cmds.setAttr("{}Shape.lineWidth".format(edge_curve), curve_width)

    if curve_spans:
        cmds.rebuildCurve(edge_curve, spans=curve_spans, ch=0, rpo=1, rt=0, end=1, kr=0, kcp=0, kep=1, kt=0, d=3, tol=0.01)

    return edge_curve


# 得到曲线上的点位置
def get_curve_position(curve_name, value):
    sel = om.MGlobal.getSelectionListByName(curve_name)
    dag_path = sel.getDagPath(0)
    m_curve = om.MFnNurbsCurve(dag_path)

    val = value
    cvs = m_curve.length()

    position_data = dict()
    for i in range(int(val)):
        param = m_curve.findParamFromLength(cvs / (val - 1) * i)
        space = om.MSpace.kWorld

        position = m_curve.getPointAtParam(param, space)
        position_data[param] = (position.x, position.y, position.z)

        #loc = cmds.spaceLocator(position = (position.x, position.y, position.z))

    return position_data


# 得到模型中心点
def get_mesh_center_point(mesh_name):
    bbox = cmds.exactWorldBoundingBox(mesh_name)
    center_x = (bbox[0] + bbox[3]) / 2
    center_y = (bbox[1] + bbox[4]) / 2
    center_z = (bbox[2] + bbox[5]) / 2

    return (center_x, center_y, center_z)


# 得到骨骼中心点
def get_jnt_middle_point(joint_name):
    selectionList = om.MSelectionList()
    selectionList.add(joint_name)
    dagPath = selectionList.getDagPath(0)

    # 获取当前骨骼的位置
    currentTransformFn = om.MFnTransform(dagPath)
    currentTranslation = currentTransformFn.translation(om.MSpace.kWorld)

    # 遍历子骨骼并计算平均位置
    childDagPaths = []
    childCount = dagPath.childCount()
    for i in range(childCount):
        child = dagPath.child(i)
        if child.hasFn(om.MFn.kJoint):
            childDagPath = om.MDagPath.getAPathTo(child)
            childDagPaths.append(childDagPath)

    totalTranslation = om.MVector(0, 0, 0)
    for childDagPath in childDagPaths:
        childTransformFn = om.MFnTransform(childDagPath)
        totalTranslation += childTransformFn.translation(om.MSpace.kWorld)

    if childDagPaths:
        averageTranslation = totalTranslation / len(childDagPaths)
    else:
        averageTranslation = currentTranslation

    # 计算中间位置
    middlePoint = (currentTranslation + averageTranslation) / 2.0
    return middlePoint


# 计算两点之间的距离
def distance_between_points(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)


# 找到最近的点位置
def find_closest_point(base_point_pos, other_points_pos):
    min_distance = float('inf')
    closest_point_pos = None

    for point_pos in other_points_pos:
        distance = math.sqrt(sum((bp - pp) ** 2 for bp, pp in zip(base_point_pos, point_pos)))
        if distance < min_distance:
            min_distance = distance
            closest_point_pos = point_pos

    return closest_point_pos


# 得到两个曲线之间最近的距离
def curve_closest_distance(curveA, curveB, accuracy=100):
    pos_data1 = get_curve_position(curveA, accuracy)
    pos_data2 = get_curve_position(curveB, accuracy)

    dist_data = dict()
    min_dist = 0
    for curve1_param, curve1_pos in pos_data1.items():
        for curve2_param, curve2_pos in pos_data2.items():
            dist = distance_between_points(curve1_pos, curve2_pos)

            dist_data[dist] = [(curve1_param, curve2_param), (curve1_pos, curve2_pos)]

            min_dist = min(dist_data.keys())
            # print(curve1_param, dist)

    min_dist_param = None
    min_dist_pos = None
    for key, value in dist_data.items():
        if min_dist == key:
            min_dist_param = value[0]
            min_dist_pos = value[1]

    # loc1 = cmds.spaceLocator(position = min_dist_pos[0])
    # loc2 = cmds.spaceLocator(position = min_dist_pos[1])

    return {"param": min_dist_param, "pos": min_dist_pos}


# 得到模型所有边的中心点
def get_edge_centers(mesh_name, ignore_below="0.000001"):
    sel = om.MGlobal.getSelectionListByName(mesh_name)
    dag = sel.getDagPath(0)
    mesh_fn = om.MFnMesh(dag)

    edge_centers = dict()
    for i in range(mesh_fn.numEdges):
        edge_vertices = mesh_fn.getEdgeVertices(i)
        point1 = mesh_fn.getPoint(edge_vertices[0], om.MSpace.kWorld)
        point2 = mesh_fn.getPoint(edge_vertices[1], om.MSpace.kWorld)

        ignore_below_num = len(ignore_below.split('.')[-1])

        center_x = (point1.x + point2.x) / 2
        ignore_center_x = float("{}.{}".format(str(center_x).split('.')[0], str(center_x).split('.')[-1][:ignore_below_num]))

        center_y = (point1.y + point2.y) / 2
        ignore_center_y = float("{}.{}".format(str(center_y).split('.')[0], str(center_y).split('.')[-1][:ignore_below_num]))

        center_z = (point1.z + point2.z) / 2
        ignore_center_z = float("{}.{}".format(str(center_z).split('.')[0], str(center_z).split('.')[-1][:ignore_below_num]))

        edge = "{}.e[{}]".format(mesh_name, i)
        edge_centers[(ignore_center_x, ignore_center_y, ignore_center_z)] = edge

    return edge_centers


# 得到模型边的两个顶点
def get_edge_vertices(mesh_name):
    sel = om.MGlobal.getSelectionListByName(mesh_name)
    dag = sel.getDagPath(0)
    mesh_fn = om.MFnMesh(dag)

    edge_vertices = dict()
    for i in range(mesh_fn.numEdges):
        vertices = mesh_fn.getEdgeVertices(i)
        edge_vertices[str(i)] = (vertices[0], vertices[1])

    return edge_vertices


# 得到边的顶点位置
def get_edge_vertices_positions(edges_list):
    vertices_positions = dict()

    for edge in edges_list:
        edge_node = pm.PyNode(edge)
        vertices = edge_node.connectedVertices()

        vertices_list = list()
        for vert in vertices:
            position = vert.getPosition(space='world')
            vertices_list.append((position.x, position.y, position.z))

        vertices_positions[edge] = vertices_list
    return vertices_positions


# 得到模型边的顶点位置
def get_mesh_vertices_positions(mesh_name):
    sel = om.MGlobal.getSelectionListByName(mesh_name)
    dag_path = sel.getDagPath(0)
    mesh_fn = om.MFnMesh(dag_path)

    vertices_positions = dict()

    # 遍历每条边
    for edge_id in range(mesh_fn.numEdges):
        # 获取边的两个顶点
        vert_ids = mesh_fn.getEdgeVertices(edge_id)

        # 获取顶点位置
        vertices_list = list()
        for vert_id in vert_ids:
            point = mesh_fn.getPoint(vert_id, om.MSpace.kWorld)
            vertices_list.append((point.x, point.y, point.z))

        vertices_positions[edge_id] = vertices_list

    return vertices_positions


# 得到模型边界边
def get_boundary_edges(mesh_name):
    sel = om.MGlobal.getSelectionListByName(mesh_name)
    dagPath = sel.getDagPath(0)
    meshIt = om.MItMeshEdge(dagPath)
    
    boundary_edges = []
    
    while not meshIt.isDone():
        # 检查当前边是否在边界上
        if meshIt.onBoundary():
            edge = "{}.e[{}]".format(mesh_name, meshIt.index())
            boundary_edges.append(edge)
    
        meshIt.next()
    
    return boundary_edges


# 将给定的两条不连续的循环边分开
def separate_edge(edges_list):
    """
    :param edges_list: 包含边缘名称的列表。
    :return: 两个边缘组的元组 (group_1, group_2)
    """
    
    # 创建一个空的MSelectionList对象
    selectionList = om.MSelectionList()

    # 将边缘添加到选择列表
    for edge in edges_list:
        selectionList.add(edge)

    # 使用OpenMaya迭代器遍历边缘
    edgeIter = om.MItSelectionList(selectionList, om.MFn.kMeshEdgeComponent)

    # 存储边缘和它们的连接顶点
    edges = {}
    while not edgeIter.isDone():
        dagPath, component = edgeIter.getComponent()
        edgeIds = om.MFnSingleIndexedComponent(component).getElements()

        # 获取边缘的顶点
        meshFn = om.MFnMesh(dagPath)
        for edgeId in edgeIds:
            vertices = meshFn.getEdgeVertices(edgeId)
            edges[edgeId] = set(vertices)

        edgeIter.next()

    # 辅助函数用于分组边缘
    def find_group(edge_id, group, edges):
        for e, verts in edges.items():
            if e in group or not verts.intersection(edges[edge_id]):
                continue
            group.add(e)
            find_group(e, group, edges)

    # 初始化两个边缘组
    group_1 = set()
    group_2 = set()

    # 分组
    for edge_id in edges:
        if edge_id in group_1 or edge_id in group_2:
            continue
        if not group_1:
            group_1.add(edge_id)
            find_group(edge_id, group_1, edges)
        elif not group_2:
            group_2.add(edge_id)
            find_group(edge_id, group_2, edges)

    return list(group_1), list(group_2)


# 根据环形曲线放样模型
def curve_loft_mesh(curve_name, thickness=0.01):
    """
    thickness: 模型的厚度参数
    """

    offset_curve1 = cmds.offsetCurve(curve_name, d=0.01, ch=False, rn=False, cb=2, st=True, cl=True, 
                                     cr=0, tol=0.01, sd=5, ugn=False, n="{}_offset1".format(curve_name))[0]
    offset_curve2 = cmds.offsetCurve(curve_name, d=-0.01, ch=False, rn=False, cb=2, st=True, cl=True, 
                                     cr=0, tol=0.01, sd=5, ugn=False, n="{}_offset1".format(curve_name))[0]

    loft_mesh = cmds.loft([offset_curve1, offset_curve2], ch=0, n="{}_LOFTMESH".format(curve_name))[0]

    poly_mesh = cmds.nurbsToPoly(loft_mesh, mnd=1, ch=0, f=0, pt=1, pc=100, chr=0.1, ft=0.01, mel=0.002, d=0.1, ut=1, un=3, 
                                 vt=1, vn=3, uch=0, ucr=0, cht=0.2, es=0, ntr=0, mrt=0, uss=1,n="{}_CUTMESH".format(curve_name))[0]


    cmds.polyMergeVertex(poly_mesh, d=0.0006, am=1, ch=0)

    loft_edges = get_boundary_edges(poly_mesh)

    loft_edge1, loft_edge2 = separate_edge(loft_edges)
    loft_edge1 = ["{}.e[{}]".format(poly_mesh, x) for x in loft_edge1]
    loft_edge2 = ["{}.e[{}]".format(poly_mesh, x) for x in loft_edge2]

    cmds.select(loft_edge1)
    cmds.MergeToCenter()
    new_loft_edges = get_boundary_edges(poly_mesh)

    cmds.polyExtrudeFacet(poly_mesh, constructionHistory=1, keepFacesTogether=1, divisions=1, twist=0, taper=1, off=0, thickness=thickness, smoothingAngle=30)
    cs_handle, cs_node = cmds.cluster(poly_mesh)
    cmds.setAttr(".s".format(cs_handle), 1.25, 1.25, 1.25)
    cmds.delete(poly_mesh, constructionHistory=True)
    cmds.delete(offset_curve1, offset_curve2, loft_mesh)
    cmds.select(cl=True)

    return poly_mesh


# 从骨骼Z轴和Y轴发射线得到离模型最近的值
def get_direction_vector(joint_name, local_direction):
    # 获取骨骼的旋转矩阵
    rotation_matrix = cmds.xform(joint_name, query=True, worldSpace=True, matrix=True)
    rotation_matrix = om.MMatrix(rotation_matrix)

    # 将局部方向向量转换为世界空间方向
    world_direction = local_direction * rotation_matrix
    return world_direction

def cast_ray_from_joint_yz(joint_name, target_mesh):
    # 获取骨骼的世界坐标
    joint_position = cmds.xform(joint_name, query=True, worldSpace=True, rotatePivot=True)
    joint_position = om.MVector(joint_position)

    # 定义局部Z轴和Y轴方向
    local_z_axis = om.MVector(0, 0, 1)
    local_neg_z_axis = om.MVector(0, 0, -1)
    local_y_axis = om.MVector(0, 1, 0)
    local_neg_y_axis = om.MVector(0, -1, 0)

    distances = []
    for local_axis in [local_z_axis, local_neg_z_axis, local_y_axis, local_neg_y_axis]:
        # 转换为世界空间方向
        world_axis = get_direction_vector(joint_name, local_axis)
        ray_direction = om.MFloatVector(world_axis.x, world_axis.y, world_axis.z)

        # 创建射线并计算距离
        selection_list = om.MSelectionList()
        selection_list.add(target_mesh)
        mesh_dag_path = selection_list.getDagPath(0)

        mesh_fn = om.MFnMesh(mesh_dag_path)
        hit_point = mesh_fn.closestIntersection(
            om.MFloatPoint(joint_position),
            ray_direction,
            om.MSpace.kWorld,
            99999,
            False
        )[0]

        distance = (om.MFloatVector(hit_point) - om.MFloatVector(joint_position)).length()
        distances.append(distance)

    # 获取距离
    distances.remove(min(distances))
    distances.remove(max(distances))

    sum_distances = sum(distances)*0.65
    return sum_distances


# 得到模型材质球
def get_material(mesh_name):
    # 尝试获取与对象关联的着色组
    shading_groups = cmds.listConnections(mesh_name, type='shadingEngine')

    # 如果直接获取不到着色组，尝试通过它的形状节点来获取
    if not shading_groups:
        shapes = cmds.listRelatives(mesh_name, shapes=True, fullPath=True) or []
        shading_groups = cmds.listConnections(shapes[0], type='shadingEngine')

    # 获取着色组的材质
    materials = cmds.ls(cmds.listConnections(shading_groups), materials=True)

    return {"shading_engine": list(set(shading_groups)), "material": list(set(materials))}


# 创建镜像模型
def mirror_mesh(mirror_list, parent_grp=None):
    for mesh in mirror_list:
        mirror_name = mesh.replace('_L', '_R').replace('_l', '_r')
        if '_R' in mesh or '_r' in mesh:
            mirror_name = mesh.replace('_R', '_L').replace('_r', '_l')
            
        dup_mesh = cmds.duplicate(mesh, n=mirror_name)
        dup_grp = cmds.group(em=True, n="{}_mirror".format(mirror_name))
        cmds.parent(dup_mesh, dup_grp)
        cmds.setAttr("{}.sx".format(dup_grp), -1)
        cmds.parent(dup_mesh, w=True)
        cmds.delete(dup_grp)
        cmds.makeIdentity(dup_mesh, apply=True, t=1, r=1, s=1, n=0, pn=1)

        if parent_grp:
            cmds.parent(dup_mesh, parent_grp)


# 修改轴心点并约束
def set_pivot_parent(root_obj, child_obj, parent=True):
    root_position = cmds.xform(root_obj, q=True, rp=True, ws=True)
    cmds.xform(child_obj, ws=True, piv=root_position)
    
    if parent:
        cmds.parentConstraint(root_obj, child_obj, mo=True, weight=1)
        # cmds.scaleConstraint(root_obj, child_obj, weight=1)


# 得到 Low 属性位置
def get_low_attr_index(obj_name):
    list_attr = cmds.listAttr(obj_name)
    is_attr = False
    for attr in list_attr:
        if attr == "body_level" or attr == "bodyLevel" or attr == "BodyLevel":    
            enum_names = cmds.attributeQuery(attr, node=obj_name, listEnum=True)
            
            if enum_names:
                enum_name_list = enum_names[0].split(':')
                # print(enum_name_list)
                
                if "Low" in enum_name_list:
                    low_index = enum_name_list.index("Low")
                
                if "low" in enum_name_list:
                    low_index = enum_name_list.index("low")

                is_attr = True
                return attr, low_index

    if not is_attr:
        cmds.addAttr(obj_name, ln="body_level", at="enum", en="Low:High:", k=True)
        return "body_level", 0


def vector_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vector_length(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def vector_normalize(v):
    length = vector_length(v)
    if length == 0:
        return [0, 0, 0]
    return [v[0] / length, v[1] / length, v[2] / length]


def vector_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# 得到骨骼指向下一个关节的轴向
def get_joint_aim_axis(joint):
    children = cmds.listRelatives(joint, type="joint", children=True)
    if not children:
        return None

    child = children[0]
    parent_pos = cmds.xform(joint, q=True, ws=True, t=True)
    child_pos = cmds.xform(child, q=True, ws=True, t=True)
    direction = vector_sub(child_pos, parent_pos)
    direction = vector_normalize(direction)

    matrix = cmds.xform(joint, q=True, m=True, ws=True)
    x_axis = vector_normalize([matrix[0], matrix[1], matrix[2]])
    y_axis = vector_normalize([matrix[4], matrix[5], matrix[6]])
    z_axis = vector_normalize([matrix[8], matrix[9], matrix[10]])

    axes = {'X': x_axis, 'Y': y_axis, 'Z': z_axis}
    max_dot = -1
    aim_axis = None
    for name, axis in axes.items():
        dot = abs(vector_dot(axis, direction))
        if dot > max_dot:
            max_dot = dot
            aim_axis = name

    return aim_axis

def check_vertex_colors(mesh_name):
    sel = om.MGlobal.getSelectionListByName(mesh_name)
    dag_path = sel.getDagPath(0)
    mfnMesh = om.MFnMesh(dag_path)
    if mfnMesh.getColorSetNames():
        return True
    
    return False


def create_poly_disc(mesh_name):
    poly_disc = cmds.polyCylinder(h=0.004, sy=1, sz=1, sx=32, ch=0, n=mesh_name)[0]
    cmds.delete([f"{poly_disc}.e[96:102]", 
                 f"{poly_disc}.e[104:110]", 
                 f"{poly_disc}.e[112:118]", 
                 f"{poly_disc}.e[120:126]",
                 f"{poly_disc}.e[128:134]",
                 f"{poly_disc}.e[136:142]",
                 f"{poly_disc}.e[144:150]",
                 f"{poly_disc}.e[152:158]"]
                 )
    cmds.polyColorPerVertex(rgb=(0.7, 0.1, 0.5), alpha=0.1, colorDisplayOption=True)
    shape = cmds.ls(poly_disc, dag=True, s=True, head=True)[0]
    cmds.setAttr(f"{shape}.overrideEnabled", 1)
    cmds.setAttr(f"{shape}.overrideColor", 21)
    
    cmds.select(cl=True)

    return(poly_disc)


def create_poly_corner(mesh_name):
    poly_corner = cmds.polyCylinder(h=0.003, sy=1, sz=1, sx=32, ch=0, n=mesh_name)[0]
    cmds.polyColorPerVertex(rgb=(0.7, 0.1, 0.5), alpha=0.1, colorDisplayOption=True)
    
    bend, bend_handle = cmds.nonLinear(poly_corner, type='bend')
    cmds.setAttr(f"{bend_handle}.rotateZ", -90)
    cmds.setAttr(f"{bend}.highBound", 0)
    cmds.setAttr(f"{bend}.curvature", -150)

    wave, wave_handle = cmds.nonLinear(poly_corner, type='wave')
    cmds.setAttr(f"{wave}.amplitude", 0.03)
    cmds.delete(poly_corner, constructionHistory=True)
    
    cmds.setAttr(f"{poly_corner}.scaleX", 1.2)
    cmds.setAttr(f"{poly_corner}.scaleY", 1.3)
    cmds.setAttr(f"{poly_corner}.rotateZ", 90)
    cmds.makeIdentity(poly_corner, apply=True, t=1, r=1, s=1, n=0, pn=1)

    shape = cmds.ls(poly_corner, dag=True, s=True, head=True)[0]
    cmds.setAttr(f"{shape}.overrideEnabled", 1)
    cmds.setAttr(f"{shape}.overrideColor", 21)
    cmds.select(cl=True)

    return(poly_corner)

