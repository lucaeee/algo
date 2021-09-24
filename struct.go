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
