package algo

import (
	"fmt"
	"sort"
)

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
		root.Total = len(root.Values)

		return res
	}

	//插入
	waitInsertNode.Values = append(waitInsertNode.Values, val)
	sort.Ints(waitInsertNode.Values)
	waitInsertNode.Total = len(waitInsertNode.Values)

	//待插入的位置是一个2节点:直接插入且排序
	if waitInsertNode.Total == 2 {

		return res
	}
	fmt.Printf("wait insert %v", waitInsertNode)
	fmt.Println("--",)

	//插入根节点为3的树
	if waitInsertNode == root && waitInsertNode.Total == 3 {

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
		fmt.Printf("cur %v", cur)
		fmt.Printf("parnent %v", curParent)
		fmt.Println()

		//提取当前节点的中键到父节点
		curParent.Values = append(curParent.Values, cur.Values[1])
		curParent.Total = len(curParent.Values)
		sort.Ints(curParent.Values)
		cur.Total = 2
		cur.Values = []int{cur.Values[0], cur.Values[2]}


		//拆分当前节点为
		var child *Node23
		//当前为叶子节点
		if len(cur.Children) == 0 {

			//构建新的有序子节点
			var tmpChildren []*Node23
			child = &Node23{Total: 1, Values: cur.Values[:1], Parent: curParent}
			cur.Values = []int{cur.Values[1]}
			fmt.Printf("child %v", child)
			for _,v := range curParent.Children {
				if v == cur {
					fmt.Printf("v-child-- %v", child)
					tmpChildren = append(tmpChildren, child)
				}
				tmpChildren = append(tmpChildren, v)
				fmt.Printf("v--- %v", v)
			}
			curParent.Children = tmpChildren
			cur.Total = len(cur.Values)
			fmt.Printf("curParent.Children %v", curParent.Children)

		}else {

			//3节点拆成俩二节点
			left  := &Node23{Total: 1, Values: cur.Values[:1], Parent: curParent}
			left.Children = []*Node23{cur.Children[0], cur.Children[1]}
			cur.Children = []*Node23{cur.Children[2], cur.Children[3]}
			cur.Values = cur.Values[1:]
			curParent.Children = append([]*Node23{left}, curParent.Children...)
			fmt.Printf("left %v", left)
			fmt.Printf("right %v", cur)
			fmt.Printf("curParent.Children %v", curParent.Children)
		}
		cur.Total = len(cur.Values)
		cur.Parent.Total = len(cur.Parent.Values)

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
	right := &Node23{Total: 1, Values: []int{cur.Values[2]}, Parent: newRoot}
	right.Children = cur.Children[2:]
	newRoot.Children = []*Node23{left, right}
	res = newRoot

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

func (node Node23) delete(val int) bool {

	//todo
	//值不存在
	//hasVal, res, parent := node.Find(val)
	//if !hasVal {
	//	return false
	//}

	//删除非叶子节点
	//删除3节点的叶子节点

	//删除2节点的叶子节点

	return  false
}

//中序遍历2-3树
func (node23 *Node23) inOrder() []int {

    var res []int
    var stack []*Node23

    //只有一个节点的树或空树
    if len(node23.Children) == 0 {

    	return node23.Values
	}
    cur := node23
	for len(stack) > 0 || cur != nil {
		for cur != nil {
			stack = append([]*Node23{cur}, stack...)
			if len(cur.Children) > 0 {
				cur = cur.Children[0]
			}else {
				cur = nil
			}
		}
		top := stack[0]
		//叶子结点
		if len(top.Children) == 0 {
			res = append(res, top.Values...)
			if len(stack) <= 1 {
				break
			}
			stack = stack[1:]
			continue
		}
		//2节点
		if len(top.Children) == 2 {
			res = append(res, top.Values...)
			if len(stack) <= 1 {
				break
			}
			stack = stack[1:]
			cur = top.Children[1]
			continue
		}
		//3节点
		if len(top.Children) == 3 {
			res = append(res, top.Values[0])
			top.Values = top.Values[1:]
			top.Children = top.Children[1:]
			cur = top.Children[0]
			continue
		}

	}
    //stack = append(stack, node23)
    //var lastPop *Node23
	//rootLeftPop := false
	//for len(stack) > 0 {
	//	cur := stack[0]
	//
	//	//根节点且左边已经出
	//	if cur == node23 && rootLeftPop {
	//		//2节点：出根入右
	//		if len(cur.Children) == 2 {
	//
	//		}else if len(cur.Children) == 3 {
	//			//3节点:出左元素删左子树
	//			rootLeftPop = false
	//		}
	//	}
	//	//叶子节点:出栈
	//	if len(cur.Children) == 0 {
	//		res = append(res, cur.Values...)
	//		lastPop = cur
	//		stack = stack[1:]
	//		if cur == node23.Children[0] {
	//			rootLeftPop = true
	//		}
	//		continue
	//	}else if  len(cur.Children) == 2 && lastPop == cur.Children[0]{
	//		//二节点且左边出栈:出栈,右节点入栈
	//	}else if len(cur.Children) == 3 && lastPop == cur.Children[0] {
	//		// 三节点且左边出栈：出栈第一个元素并且修改为2节点 右节点入栈
	//		res = append(res, cur.Values[0])
	//		cur.Values = cur.Values[1:]
	//		cur.Children = cur.Children[1:]
	//	}else {
	//
	//		//左边入栈
	//		stack = append([]*Node23{cur.Children[0]}, stack...)
	//		continue
	//	}
	//}
    return res
}



