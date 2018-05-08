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


def tree_structure(tree, dataframe, levels, datums):
    '''Single pass through the dataframe to create the tree structure.
    Works down from the root, passing already-created nodes, and
    assigns the data to the leaf node.

    Node IDs concatenate values for the different levels (joined by
    pipe characters), starting from the root.

    '''
    for row in range(len(dataframe)):
        level_list = []
        for level in levels:
            level_list.append(str(dataframe[level].iloc[row]))
        leaf_data = {}
        for datum in datums:
            leaf_data[datum] = maybefloat(dataframe[datum].iloc[row])
        for i in range(len(level_list)):
            node_id = '|'.join(level_list[:i+1])
            lunit = level_list[i]
            parent_id = '|'.join(level_list[:i])
            if tree.contains(node_id):
                pass
            elif i == 0:
                tree.create_node(node_id, node_id, data=Thile({}, lunit))
            elif i == len(level_list)-1:
                tree.create_node(node_id, node_id, parent=parent_id,
                                 data=Thile(leaf_data, lunit))
            else:
                tree.create_node(node_id, node_id, parent=parent_id,
                                 data=Thile({}, lunit))


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


def theilTree(dataframe, levels, datums):
    tree = Tree()
    tree_structure(tree, dataframe, levels, datums)
    leaf_up_tree(tree)
    return tree


########################################################################
# Functions for working with Theil trees
########################################################################


def node_weight(tree, node):
    '''Node's weight as a share of its parent'''
    return tree[node].data.total() / tree.parent(node).data.total()


def node_entropy(tree, node):
    '''Node's entropy deviation from its parent'''
    return (0 if tree.parent(node).data.entropy() == 0 else
            (tree.parent(node).data.entropy() - tree[node].data.entropy()) /
            tree.parent(node).data.entropy())


def theil_cmp(tree, node):
    '''Node's contribution to its parent's Theil statistic. Calculated as
    the node's entropy deviation from its parent, weighted by its size
    as a share of the parent.

    '''
    return node_weight(tree, node) * node_entropy(tree, node)


# Because the data are in a tree structure, one can recur through
# levels of the tree.  Note that this recursion returns zero on leaf
# nodes.  This is as it should be.  A unit with a single sub-unit
# cannot be segregated.  Letting the recursion terminate with the
# additive identity spares a lot of conditional branching to test
# whether nodes are leaves.
def theil(tree, unit, levels):
    '''Size-weighted sum of entropy deviations of a parent's children.

    '''
    the_theil = 0
    if levels == 0:
        for subunit in tree.children(unit):
            the_theil += theil_cmp(tree, subunit.identifier)
    else:
        for subunit in tree.children(unit):
            the_theil += theil_cmp(tree, subunit.identifier)
            sub_weight = subunit.data.total() / tree[unit].data.total()
            the_theil += sub_weight * theil(tree, subunit.identifier,
                                            levels - 1)
    return the_theil


def btw_theil(tree, unit):
    '''Between-child component of a parent's Theil statistic.'''
    return theil(tree, unit, 0)


# This version doesn't allow recursive Theils on the within
# component. Come back to that.
def win_theil(tree, sub_unit):
    '''Within-child component of a parent's Theil statistic.  Called on a
    specific child.

    '''
    return btw_theil(tree, sub_unit) * node_weight(tree, sub_unit)


# This assumes that the lunit in question is on level 2. Come back to
# that.
def cross_theil(tree, lunit):
    cross_theil = 0
    for node in tree.all_nodes():
        if tree[node.identifier].data.lunit == lunit:
            cross_theil += (theil_cmp(tree, node.identifier) *
                            node_weight(tree,
                                        tree.parent(node.identifier).identifier))
        else:
            pass
    return cross_theil
