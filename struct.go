package algo

import (
	"fmt"

)

type ListNode struct {
	Val  int
	Next *ListNode
	Prev *ListNode
}

func GeneralLinkList(data []int) *ListNode {

	var head *ListNode
	var last *ListNode

	for _, v := range data {

		current := &ListNode{Val: v}
		if head == nil {
			head = current
		} else {
			last.Next = current
		}
		last = current

	}

	return head
}
func (head *ListNode) getListValues() []int {
	var res []int

	current := head

	for current != nil {
		res = append(res, current.Val)
		current = current.Next
	}

	return res

}

type Stack struct {
	Val []int
}

func (stack *Stack) Push(v int) {
	stack.Val = append([]int{v}, stack.Val...)
}
func (stack *Stack) Pop() int {

	if len(stack.Val) == 0 {
		return 0
	}

	res := stack.Val[0]

	if len(stack.Val) > 1 {
		stack.Val = stack.Val[1:]
	} else {
		stack.Val = nil
	}

	return res
}
func (s *Stack) IsEmpty() bool {

	return len(s.Val) <= 0
}

func (s *Stack) Peek() int {

	if s.IsEmpty() {
		return 0
	}

	return s.Val[0]
}

type Queue struct {
	val []int
}

func (q *Queue) Push(v int) {
	q.val = append(q.val, v)
}

/*
移除首部元素
*/
func (q *Queue) Pop() {

	if len(q.val) <= 1 {
		q.val = nil
		return
	}

	q.val = q.val[1:]
}

/**
返回首部元素
**/
func (q *Queue) Peek() int {

	if len(q.val) == 0 {
		return 0
	}

	return q.val[0]
}

/**
判空
**/
func (q *Queue) Empty() bool {

	return len(q.val) <= 0
}

func (q *Queue) Size() int {

	return len(q.val)
}

/** tree **/
type BinaryTree interface {
	PrevOrderByIteration() (valList []int, nodeList []*TreeNode)
	PrevOrderByRecursive() (valList []int, nodeList []*TreeNode)

	InOrderByIteration() (valList []int, nodeList []*TreeNode)
	InOrderByRecursive() (valList []int, nodeList []*TreeNode)

	NextOrderByIteration() (valList []int, nodeList []*TreeNode)
	NextOrderByRecursive() (valList []int, nodeList []*TreeNode)
}

type TreeNode struct {
	Left  *TreeNode
	Right *TreeNode
	Val   int
}

func GeneralTree(data []int) *TreeNode {

	const empty = -999

	root := &TreeNode{Val: data[0]}
	var que []*TreeNode
	que = append(que, root)

	cur := root

	i := 1
	for cur != nil && len(que) > 0 {
		//left
		if i < len(data) && data[i] != empty {
			cur.Left = &TreeNode{Val: data[i]}
			que = append(que, cur.Left)
		}
		i++
		//right
		if i < len(data) && data[i] != empty {

			cur.Right = &TreeNode{Val: data[i]}
			que = append(que, cur.Right)
		}
		i++

		if i > len(data)-1 {
			break
		}
		que = que[1:]
		cur = que[0]
	}
	return root
}

/**
层序遍历
**/
func (root *TreeNode) LevelOrder() (valList []int, nodeList []*TreeNode, valLevel [][]int) {

	var que []*TreeNode

	que = append(que, root)

	for len(que) > 0 {
    
        var curLevel []int
        var nextQue []*TreeNode

        for i:= 0; i < len(que); i ++ {

            valList = append(valList, que[i].Val)
            nodeList = append(nodeList, que[i])
            curLevel = append(curLevel, que[i].Val)

            if que[i].Left != nil {
                nextQue = append(nextQue, que[i].Left)
            }
            if que[i].Right!= nil {
                nextQue = append(nextQue, que[i].Right)
            }

        }
        valLevel = append(valLevel, curLevel)
        que = nextQue
	}

	return valList, nodeList, valLevel
}

/**
递归前序
中左右
**/
func (root *TreeNode) PrevOrderByRecursive() (valList []int, nodeList []*TreeNode) {

	var sub func(node *TreeNode, valList *[]int, nodeList *[]*TreeNode)

	sub = func(node *TreeNode, valList *[]int, nodeList *[]*TreeNode) {

		*valList = append(*valList, node.Val)
		*nodeList = append(*nodeList, node)

		if node.Left != nil {
			sub(node.Left, valList, nodeList)
		}
		if node.Right != nil {
			sub(node.Right, valList, nodeList)
		}
	}

	sub(root, &valList, &nodeList)
	fmt.Println(valList)
	return valList, nodeList
}


/**
迭代前序
中左又
**/

func (root *TreeNode) PrevOrderByIteration() (valList []int, nodeList []*TreeNode) {
    var stack []*TreeNode

    stack = append(stack, root)

    for len(stack) > 0 {
        
        cur := stack[0]
        valList = append(valList, cur.Val)
        nodeList = append(nodeList, cur)
        
        if len(stack) > 1 {
            stack = stack[1:]
        }else {
            stack = stack[0:0]
        }

        if cur.Right != nil{

            stack = append([]*TreeNode{cur.Right}, stack...)
        }
        if cur.Left!= nil{

            stack = append([]*TreeNode{cur.Left}, stack...)
        }
    }

    return valList, nodeList
}

/**
递归中序
左中右
**/
func (root *TreeNode) InOrderByRecursive() (valList []int, nodeList []*TreeNode) {

	var sub func(node *TreeNode, valList *[]int, nodeList *[]*TreeNode)

	sub = func(node *TreeNode, valList *[]int, nodeList *[]*TreeNode) {

		if node.Left != nil {
			sub(node.Left, valList, nodeList)
		}

		*valList = append(*valList, node.Val)
		*nodeList = append(*nodeList, node)

		if node.Right != nil {
			sub(node.Right, valList, nodeList)
		}
	}

	sub(root, &valList, &nodeList)
	fmt.Println(valList)
	return valList, nodeList
}

/*
迭代中序
左中右
*/
func (root *TreeNode) InOrderByIteration() (valList []int, nodeList []*TreeNode) {
    

    var stack []*TreeNode
    
    var lastPop *TreeNode

    cur := root

    for cur != nil {

        for cur != nil {

            stack = append([]*TreeNode{cur}, stack...)
            cur = cur.Left
        }

        for len(stack) > 0 {

            cur = stack[0]
            //出栈: 左节点为空，左节点已经出栈
            if (cur.Left == nil ) || (cur.Left != nil && cur.Left == lastPop) {

                valList = append(valList, cur.Val)
                nodeList = append(nodeList, cur)
                lastPop = cur
                if len(stack) > 1 {
                    stack = stack[1:]
                }else {
                    stack = stack[0:0]
                }
            }

            //右节点重新迭代
            if cur.Right != nil{
                cur = cur.Right
                break

            }
            
        }
    }
    return valList, nodeList
}


/**
递归后序
左右中
**/
func (root *TreeNode) NextOrderByRecursive() (valList []int, nodeList []*TreeNode) {

	var sub func(node *TreeNode, valList *[]int, nodeList *[]*TreeNode)

	sub = func(node *TreeNode, valList *[]int, nodeList *[]*TreeNode) {

		if node.Left != nil {
			sub(node.Left, valList, nodeList)
		}

		if node.Right != nil {
			sub(node.Right, valList, nodeList)
		}

		*valList = append(*valList, node.Val)
		*nodeList = append(*nodeList, node)
	}

	sub(root, &valList, &nodeList)
	fmt.Println(valList)
	return valList, nodeList
}

/**
迭代后序遍历
左右中
**/
func (root *TreeNode) NextOrderByIteration() (valList []int, nodeList []*TreeNode) {
    
    var stack []*TreeNode
    
    var lastPop *TreeNode

    cur := root

    for cur != nil {

        for cur != nil {

            stack = append([]*TreeNode{cur}, stack...)
            cur = cur.Left
        }

        for len(stack) > 0 {

            cur = stack[0]
            //出栈: 叶子节点,右边节点已出，无右左出，
            if (cur.Left == nil && cur.Right == nil ) || (cur.Right != nil && cur.Right == lastPop) || (cur.Right == nil && cur.Left == lastPop) {

                valList = append(valList, cur.Val)
                nodeList = append(nodeList, cur)
                lastPop = cur
                if len(stack) > 1 {
                    stack = stack[1:]
                }else {
                    stack = stack[0:0]
                }
            }else if cur.Right != nil{
                //右节点重新迭代
                cur = cur.Right
                break

            }
            
        }
    }

    return valList, nodeList
}
