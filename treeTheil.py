from treelib import Tree
from math import log
from copy import deepcopy
import pandas as pd

########################################################################
# Functions and class for building a Theil tree
########################################################################

# The idea here is that we often calculate Theil statistics from panel
# data that has an implicit hierarchical structure. Thus for example
#
# Root, BranchA, LeafA, G1, G2
# Root, BranchA, LeafB, G1, G2
# Root, BranchB, LeafA, G1, G2
# Root, BranchB, LeafB, G1, G2
# Root, BranchC, LeafA, G1, G2
#
# ...etc. We want a dat structure that's more like a tree:
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
# for that node, and can be queried for two attributes, its total size
# and the entropy between the groups.


class Thile(object):

    def __init__(self, groups, node_level):
        self.groups = groups
        self.node_level = node_level

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
    ''' Tests whether a variable can be made into a float.  Returns a
    float if it can, the original variable if it cannot. '''
    try:
        return float(value)
    except ValueError:
        return value


def inc_dicts(fromDict, toDict):
    ''' Adds values in a pair of dictionaries. '''
    for key in toDict.keys():
        toDict[key] += fromDict[key]


def deepcopy_node_data(tree, fromNode, toNode):
    ''' Copies data structure across nodes. Need to use deepcopy rather
    than copy, because otherwise the pointer will be to the original
    data. '''
    tree[toNode].data.groups = deepcopy(tree[fromNode].data.groups)


def tree_structure(tree, dataframe, levels, datums):
    ''' Single pass through the dataframe to create the tree structure.
    Works down from the root, passing already-created nodes, and
    assigns the data to the leaf node.

    Node IDs are created by concatenating values for the different levels,
    starting from the root.  This avoids ambiguity when two nodes at the
    same level might otherwise have the same name. '''
    for row in range(len(dataframe)):
        level_list = []
        for level in levels:
            level_list.append(str(dataframe[level].iloc[row]))
        leaf_data = {}
        for datum in datums:
            leaf_data[datum] = maybefloat(dataframe[datum].iloc[row])
        for i in range(len(level_list)):
            node_id = '|'.join(level_list[:i+1])
            node_level = level_list[i]
            parent_id = '|'.join(level_list[:i])
            if tree.contains(node_id):
                pass
            elif i == 0:
                tree.create_node(node_id, node_id, data=Thile({},
                                                              node_level))
            elif i == len(level_list)-1:
                tree.create_node(node_id, node_id, parent=parent_id,
                                 data=Thile(leaf_data, node_level))
            else:
                tree.create_node(node_id, node_id, parent=parent_id,
                                 data=Thile({}, node_level))


def leaf_up_tree(tree):
    ''' Single pass through leaves to build hierarchical sums of data up
    through the tree. The leaf data dictionary is copied to the parent
    node if the latter has no data, and added to the parent node's
    data if it does. '''
    for path in tree.paths_to_leaves():
        leaf = path.pop()
        for parent in path:
            if tree[parent].data.groups == {}:
                deepcopy_node_data(tree, leaf, parent)
            else:
                inc_dicts(tree[leaf].data.groups, tree[parent].data.groups)


def theil_tree(dataframe, levels, datums):
    tree = Tree()
    tree_structure(tree, dataframe, levels, datums)
    leaf_up_tree(tree)
    return tree


########################################################################
# Functions for working with Theil statistics
########################################################################


# We often need an individual node's contribution to the Theil
# statistic, which is defined on the higher level; we tend to call
# this the "Theil statistic component."  It's the node's entropy
# deviation, weighted by its size.  To calculate it, we need the total
# and entropy properties of the node, plus those properties of its
# parent.
def theil_cmp(tree, node):
    '''Individual node's contribution to the Theil statistic one level
    higher. Calculated as the node's entropy deviation from its parent,
    weighted by its size as a share of the parent.'''
    weight = tree[node].data.total() / tree.parent(node).data.total()
    ent_dev = (0 if tree.parent(node).data.entropy() == 0 else
               (tree.parent(node).data.entropy() -
                tree[node].data.entropy()) /
               tree.parent(node).data.entropy())
    theil_cmp = weight * ent_dev
    return theil_cmp


# Now is where it gets good.  Because we have the data in a tree
# structure, we can recur through levels of the tree.  This theil()
# function does that.  Note that it returns zero on leaf nodes.  This
# is as it should be.  A unit with a single sub-unit cannot be
# segregated.  Letting it just return zeros on leaf nodes spares us a
# ton of conditional branching to test for leaves without screwing up
# the results of the calculation.

# Notice that, while the theil_cmp() function leverages the parent()
# attribute of the tree structure, the theil() function leverages the
# child() attribute.  This is where putting this information in a tree
# structure starts to pay off.  The relationship between different
# levels is explicitly accounted for by the data structure.  We don't
# need endless indices, bysorts, egens and the rest.
def theil(tree, unit, levels):
    ''' Size-weighted sum of entropy deviations of a unit's sub-units. '''
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
