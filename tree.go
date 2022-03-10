package algo

import "fmt"

type BTNode struct {
	Left  *BTNode
	Right *BTNode
	Value int
}

type BSTNode struct {
	BTNode
}

func CreateBT(values []int) *BTNode {

	if values == nil {
		return nil
	}
	size := len(values)
	var nodeList []*BTNode

	for i := 0; i < size; i++ {
		if values[i] == 0 {
			nodeList = append(nodeList, nil)
		} else {

			node := &BTNode{Value: values[i]}
			nodeList = append(nodeList, node)
		}
	}

	for i := 0; i < size; i++ {

		leftIndex := 2*i + 1
		if leftIndex < size {
			nodeList[i].Left = nodeList[leftIndex]
		}
		rightIndex := 2*i + 2
		if rightIndex < size {
			nodeList[i].Right = nodeList[rightIndex]
		}
	}

	return nodeList[0]
}

type BTIterator interface {
	PreOrderByRecursive() (nodeList []*BTNode, values []int)
	PreOrder() (nodeList []*BTNode, values []int)

	InOrderByRecursive() (nodeList []*BTNode, values []int)
	InOrder() (nodeList []*BTNode, values []int)

	PostOrderByRecursive() (nodeList []*BTNode, values []int)
	PostOrder() (nodeList []*BTNode, values []int)

	LevelOrder() (nodeList []*BTNode, values []int)
}

func (node *BTNode) PreOrderByRecursive() (nodeList []*BTNode, values []int) {

	var order func(cur *BTNode)
	order = func(cur *BTNode) {
		if cur == nil {
			return
		}

		nodeList = append(nodeList, cur)
		values = append(values, cur.Value)
		order(cur.Left)
		order(cur.Right)
	}

	order(node)

	return nodeList, values
}

func (node *BTNode) PreOrder() (nodeList []*BTNode, values []int) {
	var stack []*BTNode
	stack = append(stack, node)

	for len(stack) > 0 {
		cur := stack[0]
		nodeList = append(nodeList, cur)
		values = append(values, cur.Value)
		if len(stack) > 1 {
			stack = stack[1:]
		} else {
			stack = stack[0:0]
		}
		if cur.Right != nil {
			stack = append([]*BTNode{cur.Right}, stack...)
		}
		if cur.Left != nil {
			stack = append([]*BTNode{cur.Left}, stack...)
		}
	}

	return nodeList, values
}

func (node *BTNode) InOrderByRecursive() (nodeList []*BTNode, values []int) {

	var order func(cur *BTNode)
	order = func(cur *BTNode) {
		if cur == nil {
			return
		}

		order(cur.Left)
		nodeList = append(nodeList, cur)
		values = append(values, cur.Value)
		order(cur.Right)
	}

	order(node)

	return nodeList, values
}

func (node *BTNode) InOrder() (nodeList []*BTNode, values []int) {

	var stack []*BTNode
	cur := node

	for len(stack) > 0 || cur != nil {

		if cur != nil {
			stack = append([]*BTNode{cur}, stack...)
			cur = cur.Left
			continue
		}
		//出栈
		cur = stack[0]
		nodeList = append(nodeList, cur)
		values = append(values, cur.Value)
		if len(stack) > 1 {
			stack = stack[1:]
		} else {
			stack = stack[0:0]
		}
		//指向右节点
		cur = cur.Right
	}
	return nodeList, values
}

func (node *BTNode) PostOrderByRecursive() (nodeList []*BTNode, values []int) {

	var order func(cur *BTNode)
	order = func(cur *BTNode) {
		if cur == nil {
			return
		}

		order(cur.Left)
		order(cur.Right)
		nodeList = append(nodeList, cur)
		values = append(values, cur.Value)
	}

	order(node)

	return nodeList, values
}

func (node *BTNode) PostOrder() (nodeList []*BTNode, values []int) {

	var stack []*BTNode
	var lastPop *BTNode
	stack = append(stack, node)
	for len(stack) > 0 {
		cur := stack[0]

		//出栈：1. 叶子节点 2. 右出 3. 无右左出
		if (cur.Left == nil && cur.Right == nil) || (cur.Right != nil && cur.Right == lastPop) || (cur.Right == nil && cur.Left == lastPop) {
			nodeList = append(nodeList, cur)
			values = append(values, cur.Value)
			lastPop = cur

			if len(stack) > 1 {
				stack = stack[1:]
			} else {
				stack = stack[0:0]
			}
			continue
		}
		if cur.Right != nil {
			stack = append([]*BTNode{cur.Right}, stack...)
		}
		if cur.Left != nil {
			stack = append([]*BTNode{cur.Left}, stack...)
		}

	}
	return nodeList, values
}

func (node *BTNode) LevelOrder() (nodeList []*BTNode, values []int) {

	//bfs
	var curList []*BTNode
	curList = append(curList, node)

	for len(curList) > 0 {

		//定义在外面指针就乱了
		var nextList []*BTNode

		for _, v := range curList {
			nodeList = append(nodeList, v)
			values = append(values, v.Value)
			fmt.Println("v:", v)

			if v.Left != nil {
				nextList = append(nextList, v.Left)
			}
			if v.Right != nil {
				nextList = append(nextList, v.Right)
			}
		}
		// copy(curList, nextList)
		curList = nextList

	}

	return nodeList, values
}
