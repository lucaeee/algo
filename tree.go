package algo

import "sort"

type Node23 struct {


	Values []int
	Total int
	Children []*Node23
	Parent *Node23
}

//定义
//2节点：1个元素2个孩子或没有孩子
//3节点：2个元素3个孩子或没有孩子
//所有叶子必须都在同一层次

func NewEmptyNode23() *Node23{

	return &Node23{Total: 0}
}

func (root *Node23) Insert (val int) *Node23 {

	res := root

	var parent *Node23

	//待插入的元素已存在
	hasValue, waitInsertNode, parent := root.Find(val)
	if hasValue  {

		return res
	}

	//插入元素到空树
	if root.Total == 0 {
		root.Values = []int{val}
	}

	//插入
	waitInsertNode.Values = append(waitInsertNode.Values, val)
	sort.Ints(waitInsertNode.Values)
	waitInsertNode.Total = len(waitInsertNode.Values)

	//待插入的位置是一个2节点:直接插入且排序
	if waitInsertNode.Total == 1 {

		waitInsertNode.Total = 2

		return res
	}

	//插入根节点为3的树
	if waitInsertNode == root && waitInsertNode.Total == 2 {

		left := &Node23{Total: 1, Values: []int{waitInsertNode.Values[0]}, Parent: waitInsertNode}
		right := &Node23{Total: 1, Values: []int{waitInsertNode.Values[2]}, Parent: waitInsertNode}
		waitInsertNode.Values = []int{waitInsertNode.Values[1]}
		waitInsertNode.Total = 1
		waitInsertNode.Children = []*Node23{left, right}

		return res
	}

	//父节点为2待插入的为3:先把元素插入3节点，提取中间元素到父节点，父节点分裂成3节点
	if parent.Total == 1 && waitInsertNode.Total == 3 {

		//提取中键到父节点
		parent.Values = append(parent.Values, waitInsertNode.Values[1])
		sort.Ints(parent.Values)
		parent.Total = 2
		waitInsertNode.Values = append(waitInsertNode.Values[:1], waitInsertNode.Values[2])
		waitInsertNode.Total = 2

		//当前节点分裂出一个节点作为父节点的中节点
		mid := &Node23{Total: 1, Parent: parent}
		//当前节点位于左子树取右元素做中节点
		if waitInsertNode == parent.Children[0] {
			mid.Values = []int{waitInsertNode.Values[1]}
			waitInsertNode.Values = []int{waitInsertNode.Values[0]}
		}else{
			mid.Values = []int{waitInsertNode.Values[0]}
			waitInsertNode.Values = []int{waitInsertNode.Values[1]}
		}
		waitInsertNode.Total = 1
		parent.Children = []*Node23{parent.Children[0], mid, parent.Children[1]}

		return root
	}

	//父节点为3待插入的为3,且路径上有2节点 或从根到目标节点都是3节点
	cur := waitInsertNode
	curParent := parent
	for cur != root && cur.Total >= 3 {

		//提取当前节点的中键到父节点
		curParent.Values = append(curParent.Values, cur.Values[1])
		curParent.Total = len(curParent.Values)
		sort.Ints(curParent.Values)
		cur.Total = 2


		//拆分当前节点为
		var child *Node23
		//当前为叶子节点
		if len(cur.Children) == 0 {

			//构建新的有序子节点
			var tmpChildren []*Node23
			child = &Node23{Total: 1, Values: cur.Values[:1], Parent: curParent}
			for _,v := range curParent.Children {
				if v == cur {
					tmpChildren = append(tmpChildren, child)
				}
				tmpChildren = append(tmpChildren, v)
			}
			curParent.Children = tmpChildren
			cur.Total = 1

		}else {

			//3节点拆成俩二节点
			left  := &Node23{Total: 1, Values: cur.Values[:1], Parent: curParent}
			left.Children = []*Node23{cur.Children[0], cur.Children[1]}
			cur.Children = cur.Children[2:]
			curParent.Children = append([]*Node23{left}, curParent.Children...)
		}

		cur = cur.Parent
		curParent = cur.Parent
	}

	//递归非根节点
	if cur.Total < 3 {

		return  res
	}

	//根节点有3个元素
	newRoot := &Node23{Total: 1, Values: []int{cur.Values[1]}}
	left := &Node23{Total: 1, Values: []int{cur.Values[0]}, Parent: newRoot}
	left.Children = cur.Children[:2]
	right := &Node23{Total: 1, Values: []int{cur.Values[0]}, Parent: newRoot}
	right.Children = cur.Children[2:]
	newRoot.Children = []*Node23{left, right}
	res = root

	return  res
}


// Find 查找值是否存在
func (node *Node23)Find(val int) (res bool, cur *Node23, parent *Node23) {

	//空树
	if node == nil {

		return false, nil, nil
	}

	//命中存在的值
	for _, v := range node.Values {
		if v == val {
			return true, node, node.Parent
		}
	}

	//搜索到叶子节点，且无命中
	if len(node.Children) == 0{

		return false, node, node.Parent
	}

	//二节点
	if node.Total == 1 {

		//小于当前节点的值
		if val < node.Values[0]{

			return node.Children[0].Find(val)
		}else if val > node.Values[0]{

			//大于当前节点的值
			return node.Children[1].Find(val)
		}

	}

	//三节点
	if node.Total == 2 {

		//小于当前节点的的左元素且当前节点有左子树
		if val < node.Values[0]{

			return node.Children[0].Find(val)
		}else if val > node.Values[1]{

			//大于当前节点的的右元素且当前节点有右子树
			return node.Children[2].Find(val)
		}else if val > node.Values[0] && val < node.Values[1]{

			//介于当前元素的最大最小值且有中子树
			return node.Children[1].Find(val)
		}
	}

	return false, node, node.Parent
}


