from math import log
from copy import deepcopy
from treelib import Tree

########################################################################
# Functions and class for building a Theil tree
########################################################################

# We often calculate Theil statistics from panel data that has an
# implicit hierarchical structure. Thus for example
#
# Root, BranchA, LeafA, G1, G2
# Root, BranchA, LeafB, G1, G2
# Root, BranchB, LeafA, G1, G2
# Root, BranchB, LeafB, G1, G2
# Root, BranchC, LeafA, G1, G2
#
# ...etc. We want a data structure more like a tree:
#
# Root
#  |-BranchA
#  | |-LeafA (G1, G2)
#  | `-LeafB (G1, G2)
#  |-BranchB
#  | |-LeafA (G1, G2)
#  | `-LeafB (G1, G2)
#  `-BranchC
#    `-LeafA (G1, G2)
#
# ...etc. The Theil statistics can then be calculated recursively
# across the nodes in this tree. The "Thile" class is the data
# structure that is appended to the nodes.  It holds the group numbers
# for that node and can be queried for its total size, its lowest
# identifying unit, and the entropy between the groups.


class Thile(object):

    def __init__(self, groups, lunit):
        self.groups = groups
        self.lunit = lunit

    def total(self):
        total = sum(self.groups.values())
        return total

    def entropy(self):
        entropy = sum((0 if foo == 0 else (float(foo) / self.total() *
                                           log(self.total() /
                                               float(foo), 2))) for foo in
                      self.groups.values())
        return entropy


def maybefloat(value):
    '''Tests whether a variable can be made into a float.  Returns a
    float if it can, the original variable if it cannot.

    '''
    try:
        return float(value)
    except ValueError:
        return value


def inc_dict(fromDict, toDict):
    '''Adds values in a pair of dictionaries.'''
    for key in toDict.keys():
        toDict[key] += fromDict[key]


def deepcopy_node_data(tree, fromNode, toNode):
    '''Copies data structure across nodes. Need to use deepcopy rather
    than copy, because otherwise the pointer will be to the original
    data.

    '''
    tree[toNode].data.groups = deepcopy(tree[fromNode].data.groups)


def tree_structure(tree, csvdict, root, levels, groups):
    '''Single pass through the dataframe to create the tree. Works down
    from the root, passing already-created nodes, and assigns the data
    to the leaf node.

    Node IDs concatenate values for the different levels (joined by
    pipe characters), starting from the root.

    '''
    tree.create_node(root, root, data=Thile({}, root))
    for row in csvdict:
        nid = root
        leaf_data = {group: float(row[group]) for group in groups}
        for i in range(len(levels)):
            pid = nid
            nid += '|%s' % row[levels[i]]
            lunit = row[levels[i]]
            if tree.contains(nid):
                pass
            elif i == len(levels)-1:
                tree.create_node(nid, nid, parent=pid,
                                 data=Thile(leaf_data, lunit))
            else:
                tree.create_node(nid, nid, parent=pid, data=Thile({}, lunit))


def leaf_up_tree(tree):
    '''Single pass through leaves to hierarchically sum data up the
    tree. The child's data dictionary is deep-copied to the parent if
    the latter has no data, and added to the parent node's data if it
    does.

    '''
    for path in tree.paths_to_leaves():
        leaf = path.pop()
        for parent in path:
            if tree[parent].data.groups == {}:
                deepcopy_node_data(tree, leaf, parent)
            else:
                inc_dict(tree[leaf].data.groups, tree[parent].data.groups)


def theilTree(csvdict, root, levels, groups):
    tree = Tree()
    tree_structure(tree, csvdict, root, levels, groups)
    leaf_up_tree(tree)
    return tree


########################################################################
# Functions for working with Theil trees
########################################################################


def node_weight(tree, nid):
    '''Node's weight as a share of its parent'''
    return (0 if tree.parent(nid).data.total() == 0 else
            tree[nid].data.total() / tree.parent(nid).data.total())


def node_diversity(tree, nid):
    '''Node's diversity relative to its parent'''
    return (0 if tree.parent(nid).data.entropy() == 0 else
            tree[nid].data.entropy() / tree.parent(nid).data.entropy())


def node_entdev(tree, nid):
    '''Node's entropy deviation from its parent'''
    return (0 if tree.parent(nid).data.entropy() == 0 else
            (tree.parent(nid).data.entropy() - tree[nid].data.entropy()) /
            tree.parent(nid).data.entropy())


def node_weight_recur(tree, nid):
    '''Weight of a node in the larger tree, versus weight as a share of
    its parent.  Calculated upward to tree root.

    '''
    if tree.level(nid) == 0:
        return 1
    else:
        return node_weight(tree, nid) * \
            node_weight_recur(tree, tree.parent(nid).identifier)


def node_diversity_recur(tree, nid):
    '''Relative diversity of a node in the larger tree, versus diversity
    relative to its parent.  Calculated upward to tree root.

    '''
    if tree.level(nid) == 0:
        return 1
    else:
        return node_diversity(tree, nid) * \
            node_diversity_recur(tree, tree.parent(nid).identifier)


def theil_cmp(tree, nid):
    '''Node's contribution to its parent's Theil statistic. Calculated as
    the node's entropy deviation from its parent, weighted by its size
    as a share of the parent.

    '''
    return node_weight(tree, nid) * node_entdev(tree, nid)


# Because the data are in a tree structure, one can recur through
# levels of the tree.  Note that this recursion returns zero on leaf
# nodes.  This is as it should be.  A unit with a single sub-unit
# cannot be segregated.  Letting the recursion terminate with the
# additive identity spares a lot of conditional branching to test
# whether nodes are leaves.
def theil(tree, nid, recursions):
    '''Size-weighted sum of entropy deviations of a parent's children.

    '''
    the_theil = 0
    if recursions == 0:
        for child in tree.children(nid):
            the_theil += theil_cmp(tree, child.identifier)
    else:
        for child in tree.children(nid):
            the_theil += theil_cmp(tree, child.identifier)
            the_theil += node_weight(tree, child.identifier) * \
                node_diversity(tree, child.identifier) * \
                theil(tree, child.identifier, recursions - 1)
    return the_theil


def btw_theil(tree, nid):
    '''Between-child component of a parent's Theil statistic.'''
    return theil(tree, nid, 0)


def win_theil_cmp(tree, child_nid, recursions):
    '''Within-child component of a parent's Theil statistic.  Called on a
    specific child.  Allows recursion on the within-component Theil.

    '''
    return theil(tree, child_nid, recursions) * \
        node_weight(tree, child_nid) * node_diversity(tree, child_nid)


def win_theils(tree, nid, recursions):
    win_theils = 0
    for child in tree.children(nid):
        win_theils += win_theil_cmp(tree, child, recursions)
    return win_theils


def xwin_theil(tree, lunit):
    '''Within-child component of a tree's Theil statistic, but calculated
    on a set of nodes with a common lunit attribute rather than on a
    set of nodes that share a parent.  Automatically gives results for
    the root of the tree by recursively weighting each node's entropy
    deviation.

    It makes sense to calculate at the level of the tree root because
    this functions assumes you want to violate the tree hierarchy.  If
    you want to do such calculations for different levels, it makes
    more sense to build the tree differently.

    '''
    xwin_theil = 0
    for node in tree.filter_nodes(lambda n: n.data.lunit == lunit):
        xwin_theil += (node_entdev(tree, node.identifier) *
                       node_weight_recur(tree, node.identifier) *
                       node_diversity_recur(tree, node.identifier))
    return xwin_theil


def mtree_stage(mtree, nid):
    '''Returns the 1th element of the node ID which, in a multi-tree,
    identifies the tree within the multi-tree.

    '''
    return list(mtree[nid].identifier.split('|'))[1]


def mtree_nid(mtree, nid):
    '''Returns the 2th and subsequent elements of the node ID. '''
    return '|'.join(list(mtree[nid].identifier.split('|'))[2:])


def mtree_root(mtree, nid):
    '''Returns the 0th element of the node ID, i.e., the root.'''
    return list(mtree[nid].identifier.split('|'))[0]


def change_comps(mtree, nid):
    '''Analyzes changes in a node's Thiel component into segregation and
    population effects.  Defined on a multitree, where each major
    branch off of root is a year or other relevant stage.

    In the notation used here, E is node entropy, e is node entropy
    deviation from its parent, w is node size, and p is node weight; j
    indexes child nodes.

    '''
    Ej_new = mtree[nid].data.entropy()
    E_new = mtree.parent(nid).data.entropy()
    wj_new = mtree[nid].data.total()
    w_new = mtree.parent(nid).data.total()

    oldnid = '|'.join([mtree_root(mtree, nid),
                       str(int(mtree_stage(mtree, nid)) - 1),
                       mtree_nid(mtree, nid)])

    Ej = mtree[oldnid].data.entropy()
    E = mtree.parent(oldnid).data.entropy()
    wj = mtree[oldnid].data.total()
    w = mtree.parent(oldnid).data.total()
    pj = node_weight(mtree, oldnid)
    ej = node_entdev(mtree, oldnid)

    dotE = E_new - E
    dotEj = Ej_new - Ej
    dotwj = wj_new - wj
    dotw = w_new - w

    seg_comp = pj * ((dotE * Ej - E * dotEj)/(E * (E + dotE)))
    pop_comp = pj * (ej * ((dotwj/wj) - (dotw/w)))
    return seg_comp, pop_comp


# This is wrong, because you can't += tuples. Come back to it.
def theil_changes(tree, nid):
    theil_changes = 0
    for child in tree.children(nid):
        theil_changes += change_comps(tree, child.identifier)
    return theil_changes


# def newish_tree(dataframe, root, levels):
#     tree = Tree()
#     tree.create_node(root, root)
#     groupby_list = []
#     build_branches(tree, dataframe, root, levels, groupby_list)
#     return tree


# def build_branches(tree, dataframe, root, levels, groupby_list):
#     if levels == []:
#         return tree
#     else:
#         this_level = levels.pop()
#         if groupby_list == []:
#             for item in dataframe[this_level].unique():
#                 nid = '%s|%s' % (root, item)
#                 tree.create_node(nid, nid, parent=root)
#             groupby_list.append(this_level)
#             build_branches(tree, dataframe, root, levels, groupby_list)
#         else:
#             for group, subdf in dataframe.groupby(groupby_list):
#                 pid = root
#                 if len(groupby_list) == 1:
#                     pid += '|%s' % group
#                 else:
#                     for element in group:
#                         pid += '|%s' % element
#                 for item in subdf[this_level].unique():
#                     nid = '%s|%s' % (pid, item)
#                     if levels == []:
#                         tree.create_node(nid, nid, parent=pid,
#                                          data=subdf[subdf[this_level] == item])
#                     else:
#                         tree.create_node(nid, nid, parent=pid)
#             groupby_list.append(this_level)
#             build_branches(tree, dataframe, root, levels, groupby_list)

# def make_leaf_data(dataframe, root, levels, groups):
#     idlist = []
#     for row in range(len(dataframe)):
#         nid = root
#         lunit = dataframe[levels[-1]].iloc[row]
#         for level in levels:
#             nid += '|%s' % dataframe[level].iloc[row]
#         idlist.append([nid,
#                        {group: dataframe[group].iloc[row] for group in groups},
#                        lunit])
#     return idlist


# def add_leaf_data(tree, idlist):
#     for i in range(len(tree.leaves())):
#         tree[idlist[i][0]].data = Thile(idlist[i][1], idlist[i][2])


# def tree_structure(tree, dataframe, levels, groups):
#     '''Single pass through the dataframe to create the tree structure.
#     Works down from the root, passing already-created nodes, and
#     assigns the data to the leaf node.

#     Node IDs concatenate values for the different levels (joined by
#     pipe characters), starting from the root.

#     '''
#     i = 0
#     for row in range(len(dataframe)):
#         if i % 100000 == 0:
#             print('Did row %s' % i)
#         else:
#             pass
#         level_list = []
#         for level in levels:
#             level_list.append(str(dataframe[level].iloc[row]))
#         leaf_data = {}
#         for group in groups:
#             leaf_data[group] = maybefloat(dataframe[group].iloc[row])
#         for i in range(len(level_list)):
#             node_id = '|'.join(level_list[:i+1])
#             lunit = level_list[i]
#             parent_id = '|'.join(level_list[:i])
#             if tree.contains(node_id):
#                 pass
#             elif i == 0:
#                 tree.create_node(node_id, node_id, data=Thile({},
#                                                               lunit))
#             elif i == len(level_list)-1:
#                 tree.create_node(node_id, node_id, parent=parent_id,
#                                  data=Thile(leaf_data, lunit))
#             else:
#                 tree.create_node(node_id, node_id, parent=parent_id,
#                                  data=Thile({}, lunit))
