"""
ExpandAttackTree Model:
This model is used for Trojan detection and involves various
algorithms such as attack tree construction, matching algorithms
and marking algorithms, etc.

Author: TanShouDong.

"""

class TreeNode():
    """
    describtion :
        definition of the node  of the Expand Attack Tree
    Attribution:
        mark:
            Indicates whether the highlight is highlighted
        fun:
            Indicates the main function of the function
        weights:
            The weight of the node, including leaf nodes and internal nodes
        stv:
            Static literature index of nodes
        leaf:
            Indicates whether it is a leaf node
        type:
            Indicates whether it is a leaf node
    """
    def __init__(self):
        self.mark=None
        self.fun=""
        self.weights=0
        self.stv=0
        self.leaf=None
        self.type=""








