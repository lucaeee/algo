package algo

import (
	"fmt"
	"sort"
)

/**
qn:704
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，
写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
**/
func BinarySearch(nums []int, target int) bool {

	if len(nums) <= 0 {

		return false
	}

	leftIndex, rightIndex := 0, len(nums)-1
	fmt.Println('a')

	if target > nums[rightIndex] || target < nums[leftIndex] {

		return false
	}

	for {

		mid := (leftIndex + rightIndex) / 2

		if nums[mid] == target {

			return true
		} else if rightIndex == leftIndex || rightIndex-1 == leftIndex {

			if nums[leftIndex] == target || nums[rightIndex] == target {

				return true
			} else {

				return false
			}

		} else if nums[mid] > target {

			rightIndex = mid
		} else if nums[mid] < target {

			leftIndex = mid
		}
	}

	return false
}

/**
qn:27
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
**/
func removeElement(nums []int, val int) int {

	length := len(nums)
	if length == 0 {

		return -1
	}

	left, right := 0, length-1

	for {

		if left == right {

			if length == 1 {

				if nums[left] == val {

					return 0
				} else {

					return 1
				}
			} else {

				return left + 1
			}
		}
		if nums[left] == val {

			if nums[right] == val {

				right--
			} else {

				nums[left], nums[right] = nums[right], nums[left]
				left++
				right--
			}
		} else {

			left++
		}
	}
}

/**
不改变顺序
**/
func removeElement2(nums []int, val int) int {

	length := len(nums)

	//指针重合时指向的是待删除的元素
	slow := 0

	for fast := 0; fast < length; fast++ {

		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
		}
	}

	return slow
}

/**
qn:977
给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]
**/
func sortedSquares(nums []int) []int {

	for k, v := range nums {

		if v < 0 {
			nums[k] *= -1
		}
	}
	sort.Ints(nums)

	var res []int

	for _, v := range nums {

		res = append(res, v*v)
	}

	return res
}

//从最大值开始
func sortedSquares2(nums []int) []int {

	left, right := 0, len(nums)-1

	var res []int

	for {
		leftVal := nums[left] * nums[left]
		rightVal := nums[right] * nums[right]

		if left == right {

			res = append([]int{rightVal}, res...)
			break
		}
		if rightVal >= leftVal {

			res = append([]int{rightVal}, res...)
			right--
		} else {
			res = append([]int{leftVal}, res...)
			left++
		}
	}

	return res
}

/**
qn:209
给定一个含有 n 个正整数的数组和一个正整数 target 。
找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。
如果不存在符合条件的子数组，返回 0 。

输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
**/
func minSubArrayLen(target int, nums []int) int {

	var sum, left, minSub, right = 0, 0, 0, 0

	for left = 0; left < len(nums); left++ {

		sum = 0
		for right = left; right < len(nums); right++ {

			sum += nums[right]
			if sum >= target {

				sub := right - left + 1
				sum = 0
				//第一次查找到符合条件的子串
				if minSub == 0 {
					minSub = sub
				} else {
					//当前子串比目前已经找到的子串少
					if sub < minSub {
						minSub = sub
					}
				}

			}

		}
	}

	return minSub
}

func minSubArrayLen2(target int, nums []int) int {

	var sum, left, minSub, right = 0, 0, 0, 0

	for right = 0; right < len(nums); right++ {
		sum += nums[right]

		//查找到符合条件的子串, 1.当r=length的时候不会跳出
		for sum >= target {

			sub := right - left + 1

			if minSub == 0 || sub < minSub {
				minSub = sub
			}

			//左指针加1且和去掉当前左指针的值
			sum -= nums[left]
			left++
		}

	}
	return minSub
}

/**
59. 螺旋矩阵 II
给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
**/

func generateMatrix(n int) [][]int {

	var res [][]int
	//init
	for i := 1; i <= n; i++ {

		tmp := make([]int, n)
		res = append(res, tmp)
	}

	var xMax, yMax = n - 1, n - 1
	var xMin, yMin = 0, 0

	var x, y = 0, 0

	i := 1

	for i <= n*n {

		//左上角顶点 左右
		if x == xMin && y == yMin {

			for y <= yMax {
				fmt.Println(i)
				res[x][y] = i
				y++
				i++
			}
			// xMin++
			y--
		}
		//有上角顶点 上下
		if y == yMax && x == xMin {
			xMin++
			x = xMin
			for x <= xMax {
				fmt.Println(i)
				res[x][y] = i
				i++
				x++
			}
			x--
		}
		//右下角顶点  右左
		if y == yMax && x == xMax {

			yMax--
			y = yMax
			for y <= yMax && y >= yMin {
				fmt.Println(i)
				res[x][y] = i
				i++
				y--
			}

			y++
		}
		//左下角顶点 下上
		if x == xMax && y == yMin {

			xMax--
			x = xMax
			for x <= xMax && x >= xMin {
				fmt.Println(i)
				res[x][y] = i
				i++
				x--
			}
			yMin++
			y = yMin
			x++
		}

	}
	fmt.Println(res)
	return res
}

/**
qn:203
给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]
**/
func removeElements(head *ListNode, val int) *ListNode {

	var newHead, lastNotEqualNode *ListNode

	current := head

	for current != nil {

		if current.Val == val {

			if lastNotEqualNode != nil {

				lastNotEqualNode.Next = current.Next
			}
		} else {

			if lastNotEqualNode == nil {
				newHead = current
			}
			lastNotEqualNode = current
		}

		current = current.Next
	}

	return newHead
}

/*
206. 反转链表
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
修改后继接地那
*/
func reverseList(head *ListNode) *ListNode {

	var prev, newHead *ListNode

	prev, current, newHead := nil, head, nil

	for current != nil {
		beforeNext := current.Next
		current.Next = prev

		newHead = current

		prev = current
		current = beforeNext
	}

	return newHead
}

/**
24. 两两交换链表中的节点
给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换
1234
2143 todo
**/
func swapPairs(head *ListNode) *ListNode {
	//占位头节点
	virtualHead := &ListNode{
		Next: head,
	}
	// pre := virtualHead

	return virtualHead.Next

}

/**
qn:19. 删除链表的倒数第 N 个结点
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

进阶：你能尝试使用一趟扫描实现吗？
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
**/
func removeNthFromEnd(head *ListNode, n int) *ListNode {

	fast := 1
	slowNode, fastNode := head, head

	var pre *ListNode
	for ; fast < n; fast++ {
		fastNode = fastNode.Next
	}

	for fastNode.Next != nil {

		pre = slowNode
		slowNode = slowNode.Next
		fastNode = fastNode.Next
	}

	//删除头节点
	if pre == nil {

		return head.Next
	} else {

		if fastNode != slowNode {
			pre.Next = fastNode
		} else {
			pre.Next = nil
		}
		return head
	}
}

func removeNthFromEnd2(head *ListNode, n int) *ListNode {

	virtualHead := &ListNode{Next: head}

	slowNode, fastNode := virtualHead, virtualHead

	i := 0
	for i = 0; fastNode.Next != nil; i++ {

		if i >= n {
			slowNode = slowNode.Next
		}
		fastNode = fastNode.Next
	}

	if n <= i {

		slowNode.Next = slowNode.Next.Next
	}
	return virtualHead.Next
}

/**
面试题 02.07. 链表相交
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
**/

func GetIntersectionNode(headA, headB *ListNode) *ListNode {
	headAmap := map[*ListNode]bool{}
	var res *ListNode

	a := headA

	for a != nil {
		headAmap[a] = true
		a = a.Next
	}

	b := headB
	for b != nil {
		_, ok := headAmap[b]

		if ok {
			res = b
			break
		}
		b = b.Next
	}

	return res
}

//todo
