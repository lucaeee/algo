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
