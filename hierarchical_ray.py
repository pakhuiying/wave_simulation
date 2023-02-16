from collections import deque #to implement stack

class Node:
    """
    construct a binary tree from the indices of daughter_ray
    child_key (str)
    """
    def __init__(self,key):
        self.reflected = None
        self.transmitted = None
        self.key = key
    def insert_node(self, key):
        if self.key == key[:-2]:
            if key.endswith('_r') and self.reflected is None:
                self.reflected = Node(key)
            elif key.endswith('_t') and self.transmitted is None:
                self.transmitted = Node(key)
        else:
            if self.key + '_r' in key:
                self.reflected.insert_node(key)
            elif self.key + '_t' in key:
                self.transmitted.insert_node(key)
    
    def print_nodes(self):
        if self.reflected:
            self.reflected.print_nodes()
        print(self.key)
        if self.transmitted:
            self.transmitted.print_nodes()
            
class Tree:
    """
    maintains the hierarchical structure of multiple scattering from a parent ray
    prent_ray has a digit key
    """
    def __init__(self, node):
        self.node = node
        self.root = None
    def get_parent_node(self,node):
        if node.key.isdigit():
            self.root = node
            return node # returns a Node
        else:
            parent_node = Node(node.key[:-2])
            if node.key.endswith('_r'):
                parent_node.reflected = node
            else:
                parent_node.transmitted = node
            
            return self.get_parent_node(parent_node)
    
    def print_nodes(self):
        if self.root:
            r = self.root
            stack = deque()
            L = [r.key]
            stack.append(r)
            
            while len(stack) > 0:
                node = stack.pop()
                if node.reflected is not None:
                    stack.append(node.reflected)
                    L.append(node.reflected.key)
                if node.transmitted is not None:
                    stack.append(node.transmitted)
                    L.append(node.transmitted.key)
            return L
        else:
            return None
    
class Forest:
    """
    inputs are keys of daughter rays
    >>> grow_trees('2_r_t')
    """
    def __init__(self):
        self.tree_list = {}
    
    def grow_trees(self,key):
        if len(self.tree_list) == 0:
            node = Node(key)
            T = Tree(node)
            parent_node = T.get_parent_node(node)
            self.tree_list[parent_node.key] = parent_node
        else:
            create_tree = True
            for k,v in self.tree_list.items():
                if k in key:
                    self.tree_list[k].insert_node(key)
                    create_tree = False

            if create_tree is True:
                node = Node(key)
                T = Tree(node)
                parent_node = T.get_parent_node(node)
                self.tree_list[parent_node.key] = parent_node
    
    def print_nodes(self):
        if bool(self.tree_list):
            for k,v in self.tree_list.items():
                print(f'parent ray: {k}')
                v.print_nodes()