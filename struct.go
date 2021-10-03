package algo

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
func (stack *Stack) Push (v int) {
    stack.Val = append([]int{v}, stack.Val...)
}
func (stack *Stack) Pop() int {

    if len(stack.Val) == 0 {
       return 0 
    }

    res := stack.Val[0] 
    
    if len(stack.Val) > 1 {
        stack.Val = stack.Val[1:]
    }else {
        stack.Val = nil
    }

    return res
}
func (s *Stack) IsEmpty() bool {

    return len(s.Val) > 0 
} 


type Queue struct {
    val []int
}

func (q *Queue) Push (v int) {
    q.val = append(q.val, v)
}
/*
移除首部元素
*/
func (q *Queue) Pop () {
    
    if len(q.val) <= 1 {
        q.val = nil
        return
    }

    q.val = q.val[1:]
    return
}
/**
返回首部元素
**/
func(q *Queue) Peek () int {
    
    if len(q.val) == 0 {
        return 0
    }

    return q.val[0]
}
/**
判空
**/
func(q *Queue) Empty() bool {
    
    return len(q.val) <= 0
}

func (q *Queue) Size() int {

    return len(q.val)
}
