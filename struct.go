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
