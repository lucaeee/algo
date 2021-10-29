package algo

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
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

/**
qn:142
环形链表 II
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
说明：不允许修改给定的链表。
进阶：
你是否可以使用 O(1) 空间解决此题？
**/
func DetectCycle(head *ListNode) *ListNode {
	//hash法
	var res *ListNode
	nodeMap := map[*ListNode]bool{}
	current := head

	for current != nil {

		_, ok := nodeMap[current]
		if ok {
			res = current
			break
		} else {
			nodeMap[current] = true
			current = current.Next
		}
	}

	return res
}

//todo 双指针

/**----hash-----***/

/**
qn:242. 有效的字母异位词
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。
输入: s = "anagram", t = "nagaram"
输出: true

输入: s = "rat", t = "car"
输出: false
**/
func IsAnagram(s string, t string) bool {

	res := true

	sBytes := []byte(s)
	tBytes := []byte(t)
	sMap, tMap := make(map[byte]int), make(map[byte]int)
	for _, v := range sBytes {

		count, ok := sMap[v]
		if ok {
			sMap[v] = count + 1
		} else {
			sMap[v] = 1
		}
	}
	for _, v := range tBytes {

		count, ok := tMap[v]
		if ok {
			tMap[v] = count + 1
		} else {
			tMap[v] = 1
		}
	}

	for k, v := range sMap {

		count, ok := tMap[k]

		if !ok || count != v {
			res = false
			break
		}
	}

	return res
	//法2 加1减1
}

//todo 1002. 查找常用字符 看不懂题目

/**
qn:349. 两个数组的交集
给定两个数组，编写一个函数来计算它们的交集。
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]
**/
func Intersection(nums1 []int, nums2 []int) []int {

	var res []int
	var resMap = make(map[int]bool)
	var nums1Map = make(map[int]bool)

	for _, v := range nums1 {
		nums1Map[v] = true
	}

	for _, v := range nums2 {

		_, ok := nums1Map[v]
		if ok {
			resMap[v] = true
		}
	}

	for k, _ := range resMap {
		res = append(res, k)
	}

	return res
}

/**
qn:202. 快乐数
编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」定义为：

对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
如果 可以变为  1，那么这个数就是快乐数。
如果 n 是快乐数就返回 true ；不是，则返回 false 。
输入：19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

输入：n = 2
输出：false
**/
func IsHappy(n int) bool {
	m := make(map[int]bool)

	getSum := func(n int) int {
		sum := 0
		for n > 0 {
			sum += (n % 10) * (n % 10)
			n = n / 10
		}
		fmt.Println(sum)
		return sum
	}
	//如果不是的话 sum的值会和之前某个数一样
	for n != 1 && !m[n] {
		// fmt.Println(m[n])
		n, m[n] = getSum(n), true
	}

	return n == 1
}

/**
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

**/
func TwoSum(nums []int, target int) []int {

	numsMap := make(map[int]int)

	for k, v := range nums {
		numsMap[v] = k
	}

	var res []int

	for k, v := range numsMap {

		poor := target - k

		pv, pk := numsMap[poor]
		if pk && pv != v {
			res = append(res, v)
			res = append(res, pv)
			break
		}

	}

	return res
}

/**
qn:454. 四数相加 II
给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。

为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -228 到 228 - 1 之间，最终结果不会超过 231 - 1 。

输入:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

输出:
2
次数次数--不用考虑重复的情况

解释:
两个元组如下:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
**/
func FourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {

	m := make(map[int]int)
	count := 0
	for _, v1 := range nums1 {
		for _, v2 := range nums2 {
			m[v1+v2]++
		}
	}
	for _, v3 := range nums3 {
		for _, v4 := range nums4 {
			count += m[-v3-v4]
		}
	}
	return count
}

/**

给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。

(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)

示例 1：

输入：ransomNote = "a", magazine = "b"
输出：false
示例 2：

输入：ransomNote = "aa", magazine = "ab"
输出：false
示例 3：

输入：ransomNote = "aa", magazine = "aab"
输出：true
*
**/
func CanConstruct(ransomNote string, magazine string) bool {

	mBytes := []byte(magazine)
	rBytes := []byte(ransomNote)

	mMap := make(map[byte]int)

	res := true
	for _, v := range mBytes {

		mMap[v]++
	}

	for _, v := range rBytes {

		if mMap[v] <= 0 {
			res = false
			break
		} else {
			mMap[v]--
		}
	}

	return res
}

/**
qn:15. 三数之和

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

示例 1：
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
示例 2：
输入：nums = []
输出：[]
示例 3：
输入：nums = [0]
输出：[]
**/
func ThreeSum(nums []int) [][]int {

	var res [][]int

	resMap := make(map[string]int)

	numsMap := make(map[int]int)

	if len(nums) < 3 {
		return res
	}

	//记录同一个值出现的次数
	for _, v := range nums {

		numsMap[v]++
	}

	//组合出a+b所有可能的值
	for i := 0; i < len(nums); i++ {
		nums[i], nums[0] = nums[0], nums[i]
		for j := 1; j < len(nums); j++ {

			poor := nums[0] + nums[j]
			numsMap[nums[0]]--
			numsMap[nums[j]]--
			if numsMap[-poor] >= 1 {
				tmp := []int{-poor, nums[0], nums[j]}
				sort.Ints(tmp)
				js, _ := json.Marshal(tmp)
				//结果集中不存在
				if resMap[string(js)] <= 0 {
					resMap[string(js)] = 1
					res = append(res, tmp)
				}
			}
			numsMap[nums[0]]++
			numsMap[nums[j]]++

		}
		nums[i], nums[0] = nums[0], nums[i]
	}
	fmt.Println(res)
	return res
}

//todo 排序双指针

/**
qn:18. 四数之和
给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] ：

0 <= a, b, c, d < n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target
你可以按 任意顺序 返回答案 。

示例 1：
输入：nums = [1,0,-1,0,-2,2], target = 0
输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

示例 2：
输入：nums = [2,2,2,2,2], target = 8
输出：[[2,2,2,2]]
**/
func FourSum(nums []int, target int) [][]int {
	var res [][]int

	if len(nums) <= 3 {
		return res
	}

	//升序排序
	sort.Ints(nums)
	// fmt.Println(nums)
	for n1 := 0; n1 <= len(nums)-4; n1++ {

		//两个元素相等直接走下一个
		if n1 > 0 && nums[n1] == nums[n1-1] {

			continue
		}
		for n2 := n1 + 1; n2 <= len(nums)-3; n2++ {

			for left, right := n2+1, len(nums)-1; left != right; {

				sum := nums[n1] + nums[n2] + nums[left] + nums[right]
				if sum < target {
					left++
					if nums[left] == nums[left-1] {
						continue
					}
				} else if sum > target {
					right--
					if nums[right] == nums[right-1] {
						continue
					}
				} else if sum == target {
					//找到相等的元素
					tmp := []int{nums[n1], nums[n2], nums[left], nums[right]}
					fmt.Println(tmp)
					res = append(res, tmp)
					break
				}
			}
		}
	}

	return res
}

/***-----string----**/

/*
qn:344. 反转字符串
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
示例 1：
输入：s = ["h","e","l","l","o"]
输出：["o","l","l","e","h"]
示例 2：
输入：s = ["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]
*/
func ReverseString(s []byte) {

	if len(s) <= 1 {

		return
	}

	left, right := 0, len(s)-1

	for left != right {

		s[left], s[right] = s[right], s[left]

		left++
		right--
	}

	fmt.Println(string(s))
}

/**
qn:541. 反转字符串 II
给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。

如果剩余字符少于 k 个，则将剩余字符全部反转。
如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。


示例 1：

输入：s = "abcdefg", k = 2
输出："bacdfeg"
示例 2：

输入：s = "abcd", k = 2
输出："bacd"
**/
func ReverseStr(s string, k int) string {

	var res string
	sBytes := []byte(s)
	//k =1 没意义不判断

	rev := func(s []byte, left int, right int) {

		for left < right {
			s[left], s[right] = s[right], s[left]
			left++
			right--
		}
	}

	start := 0
	for i := 0; i < len(sBytes); i++ {

		if (i+1)%(2*k) == 0 {
			//反转前k个
			rev(sBytes, start, start+k-1)
			start = i + 1
		}
	}

	//剩余字符个数
	remain := len(sBytes) - start
	//剩余字符小于k个
	if remain < k {

		rev(sBytes, start, len(sBytes)-1)
	} else if remain >= k {
		rev(sBytes, start, start+k-1)
	}
	res = string(sBytes)
	fmt.Println(res)
	return res
	//优解让 i += (2 * k)，i 每次移动 2 * k 就可以了
}

/**
剑指 Offer 05. 替换空格
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

示例 1：

输入：s = "We are happy."
输出："We%20are%20happy."

限制：
0 <= s 的长度 <= 10000
**/
func ReplaceSpace(s string) string {

	b := []byte(s)
	result := make([]byte, 0)
	for i := 0; i < len(b); i++ {
		if b[i] == ' ' {
			result = append(result, []byte("%20")...)
		} else {
			result = append(result, b[i])
		}
	}
	return string(result)
	//2. 扩容后从后到前替换
}

/**
151. 翻转字符串里的单词
给你一个字符串 s ，逐个翻转字符串中的所有 单词 。
单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。
说明：

输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
翻转后单词间应当仅用一个空格分隔。
翻转后的字符串中不应包含额外的空格。

输入：s = "  Bob    Loves  Alice   "
输出："Alice Loves Bob"

提示：
1 <= s.length <= 104
s 包含英文大小写字母、数字和空格 ' '
s 中 至少存在一个 单词
**/
func ReverseWords(s string) string {

	//[66 111 98 32 32 32 32 76 111 118 101 115 32 32 65 108 105 99 101 32 32 32]
	// sBytes := []byte(s)

	sBytes := []byte(s)
	// fmt.Println(sBytes)
	slow, fast := 0, 0

	//fast指向第一个不为空的字符
	for sBytes[fast] == 32 && fast < len(sBytes) {
		fast++
	}
	//去除多余空格，整体前移
	for fast < len(sBytes) {

		//如果当前元素为空格且当前元素的前一个元素也为空则不用赋值：上一次循环已经赋值过一个空格了
		if fast > 1 && sBytes[fast] == 32 && sBytes[fast] == sBytes[fast-1] {
			//慢指针不动，快指针加1
			fast++
			// fmt.Println(fast)
			continue
		}
		sBytes[slow] = sBytes[fast]
		slow++
		fast++
	}
	//去除尾部空格
	if sBytes[slow-1] == 32 {
		slow--
	}
	sBytes = sBytes[:slow]

	reverse := func(sBytes []byte, left int, right int) {

		for left < right {

			sBytes[left], sBytes[right] = sBytes[right], sBytes[left]
			left++
			right--
		}
	}

	//整个数组反转
	reverse(sBytes, 0, len(sBytes)-1)
	//反转每个空格的元素
	start, end := 0, 0
	for ; end < len(sBytes); end++ {

		//反转
		if sBytes[end] == 32 {
			reverse(sBytes, start, end-1)
			start = end + 1

		}
		//最后一次没有空格做标记需要再反转一次
		if end == len(sBytes)-1 {
			reverse(sBytes, start, end)
		}
	}

	return string(sBytes)
}

/**
剑指 Offer 58 - II. 左旋转字符串
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

示例 1：

输入: s = "abcdefg", k = 2
输出: "cdefgab"
示例 2：

输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
**/
func ReverseLeftWords(s string, n int) string {

	sBytes := []byte(s)

	if len(sBytes) <= 1 || len(sBytes) <= n {
		return s
	}

	reverse := func(sBytes []byte, left int, right int) {

		for left < right {

			sBytes[left], sBytes[right] = sBytes[right], sBytes[left]
			left++
			right--
		}
	}
	reverse(sBytes, 0, len(sBytes)-1)

	reverse(sBytes, 0, len(sBytes)-n-1)
	reverse(sBytes, len(sBytes)-n, len(sBytes)-1)
	return string(sBytes)
}

/**
28. 实现 strStr()
实现 strStr() 函数。

给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。


说明：

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。


示例 1：

输入：haystack = "hello", needle = "ll"
输出：2
示例 2：

输入：haystack = "aaaaa", needle = "bba"
输出：-1
示例 3：

输入：haystack = "", needle = ""
输出：0
**/
func strStr(haystack string, needle string) int {
	//todo
	return 1
}

/**
qn:459. 重复的子字符串
给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。

示例 1:
输入: "abab"
输出: True
解释: 可由子字符串 "ab" 重复两次构成。

示例 2:
输入: "aba"
输出: False
示例 3:

输入: "abcabcabcabc"
输出: True
解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
**/
func RepeatedSubstringPattern(s string) bool {

	sBytes := []byte(s)

	for i := 1; i < len(sBytes)-1; i++ {
		j := 0
		fmt.Printf("sub %v", string(sBytes[:i]))
		for k := i; k <= len(sBytes)-1; k++ {
			//元素不相等
			if sBytes[j] != sBytes[k] {
				break
			} else {
				if k == len(sBytes)-1 {
					fmt.Printf("|success %v", string(sBytes[:i]))
					return true
				}
			}
			j++
			if j == i {
				j = 0
			}
		}
	}

	return false
}

/**

232. 用栈实现队列
请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

实现 MyQueue 类：

void push(int x) 将元素 x 推到队列的末尾
int pop() 从队列的开头移除并返回元素
int peek() 返回队列开头的元素
boolean empty() 如果队列为空，返回 true ；否则，返回 false


说明：

你只能使用标准的栈操作 —— 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。


进阶：

你能否实现每个操作均摊时间复杂度为 O(1) 的队列？换句话说，执行 n 个操作的总时间复杂度为 O(n) ，即使其中一个操作可能花费较长时间。


示例：

输入：
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 1, 1, false]

解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false


提示：

1 <= x <= 9
最多调用 100 次 push、pop、peek 和 empty
假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）

**/
type MyQueue struct {
	In  *Stack
	Out *Stack
}

//func Constructor() MyQueue {

//}

func (myQueue *MyQueue) Push(x int) {
	myQueue.In.Push(x)
}

func (myQueue *MyQueue) Pop() int {

	if myQueue.In.IsEmpty() && myQueue.Out.IsEmpty() {

		return 0
	} else if myQueue.Out.IsEmpty() {

		val := myQueue.In.Pop()
		myQueue.Out.Push(val)
	}

	return myQueue.Out.Pop()
}

/*返回开头元素*/
func (myQueue *MyQueue) Peek() int {

	return 1
}

func (myQueue *MyQueue) Empty() bool {

	return false
}

/**
20. 有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。

输入：s = "()"
输出：true

输入：s = "()[]{}"
输出：true

输入：s = "(]"
输出：false

输入：s = "([)]"
输出：false

输入：s = "{[]}"
输出：true
**/
func IsValid(s string) bool {

	stack := Stack{}
	sBytes := []byte(s)

	stack.Push(int(sBytes[0]))
	peek := stack.Peek()

	leftMap := make(map[byte]bool)
	rightMap := make(map[byte]bool)

	for _, v := range []byte("{([") {
		leftMap[v] = true
	}
	for _, v := range []byte("})]") {
		rightMap[v] = true
	}

	if rightMap[byte(peek)] {
		return false
	}
	match1 := []byte("{}")
	match2 := []byte("[]")
	match3 := []byte("()")

	for i := 1; i < len(sBytes); i++ {

		if leftMap[sBytes[i]] {
			stack.Push(int(sBytes[i]))
		} else {

			cur := []byte{byte(peek), sBytes[i]}
			if bytes.Equal(match1, cur) || bytes.Equal(match2, cur) || bytes.Equal(match3, cur) {
				stack.Pop()
				continue
			} else {
				return false
			}
		}
		if stack.IsEmpty() {
			return false
		} else {

			peek = stack.Peek()
		}
	}

	return true
}

/**
qn:1047. 删除字符串中的所有相邻重复项
给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

示例：

输入："abbaca"
输出："ca"
**/
func RemoveDuplicates(s string) string {

	sBytes := []byte(s)

	stack := Stack{}
	stack.Push(int(sBytes[0]))

	peek := stack.Peek()

	for i := 1; i < len(sBytes); i++ {

		if peek != int(sBytes[i]) {
			stack.Push(int(sBytes[i]))
			peek = stack.Peek()
		} else {
			stack.Pop()
			peek = stack.Peek()
		}
	}

	var res []byte

	//遍历要倒着来
	for i := len(stack.Val) - 1; i >= 0; i-- {

		res = append(res, byte(stack.Val[i]))
	}

	return string(res)
}

/**
qn:150. 逆波兰表达式求值
根据 逆波兰表示法，求表达式的值。

有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

说明：

整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：
该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
**/
func EvalRPN(tokens []string) int {

	stack := Stack{}
	first, _ := strconv.Atoi(tokens[0])
	stack.Push(first)
	for i := 1; i < len(tokens); i++ {
		fmt.Println("stack", stack)
		if tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/" {

			opRes := 0
			left := stack.Peek()
			stack.Pop()
			right := stack.Peek()
			stack.Pop()
			switch tokens[i] {
			case "+":
				opRes = left + right
			case "-":
				opRes = left - right
			case "*":
				opRes = left * right
			case "/":
				opRes = right / left

			}
			stack.Push(opRes)
		} else {
			first, _ := strconv.Atoi(tokens[i])
			stack.Push(first)
		}

	}
	fmt.Println("stack2", stack)
	return stack.Peek()
}

/*
239. 滑动窗口最大值
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
*/
func MaxSlidingWindow(nums []int, k int) []int {
	var res []int

	// windownLeft, windownRight := 0, k-1
	// var stack []int
	// lastMax := 0

	// for windownLeft < len(nums) && windownRight < len(nums) {

	// 	var windownMax int

	// 	if len(stack) == 0 {

	// 		for cur := windownRight; cur >= windownLeft; cur-- {
	// 			if nums[cur] >= windownMax {
	// 				windownMax = nums[cur]
	// 			}
	// 			stack = append([]int{nums[cur]}, stack...)
	// 		}

	// 	} else {

	// 		pop := stack[0]
	// 		stack = stack[1:]
	// 		stack = append([]int{nums[windownRight]}, stack...)

	// 		if nums[windownRight] >= lastMax {
	// 			windownMax = nums[windownRight]
	// 		} else {
	// 			//上一轮的最大值出栈
	// 			if pop == lastMax {
	// 				wait := make([]int, k)
	// 				copy(wait, nums[windownLeft:windownRight])
	// 				sort.Ints(wait)
	// 				windownMax = wait[len(wait)-1]
	// 			} else {
	// 				windownMax = lastMax
	// 			}
	// 		}
	// 	}
	// 	lastMax = windownMax
	// 	res = append(res, windownMax)
	// 	windownLeft++
	// 	windownRight++
	// }

	//左出又进
	var stack []int

	//丢弃小的元素
	push := func(s *[]int, v int) {
		for len(*s) > 0 && v > (*s)[len(*s)-1] {
			*s = (*s)[:len(*s)-1]
		}
		*s = append(*s, v)
	}

	//出队:窗口左边相邻的值且是最大值
	pop := func(s *[]int, v int) {

		if len(*s) > 0 && v == (*s)[0] {
			*s = (*s)[1:]
		}
	}

	for i := 0; i < k; i++ {
		push(&stack, nums[i])
	}
	res = append(res, stack[0])
	for j := k; j < len(nums); j++ {
		pop(&stack, nums[j-k])
		push(&stack, nums[j])
		res = append(res, stack[0])

		fmt.Println(stack)
	}

	return res
}

/*
347. 前 K 个高频元素
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
示例 2:

输入: nums = [1], k = 1
输出: [1]

提示：

1 <= nums.length <= 105
k 的取值范围是 [1, 数组中不相同的元素的个数]
题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的


进阶：你所设计算法的时间复杂度 必须 优于 O(n log n) ，其中 n 是数组大小。
*/
// func TopKFrequent(nums []int, k int) []int {
// 	//todo
// 	var res []int

// 	return res
// }

/**
226. 翻转二叉树
翻转一棵二叉树。

示例：

输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1
**/

func InvertTree(root *TreeNode) *TreeNode {

	var sub func(node *TreeNode)

	sub = func(node *TreeNode) {

		if node != nil {

			node.Left, node.Right = node.Right, node.Left

			if node.Left != nil {

				sub(node.Left)
			}
			if node.Right != nil {

				sub(node.Right)
			}
		}

	}

	sub(root)

	return root
}

/**
101. 对称二叉树
给定一个二叉树，检查它是否是镜像对称的。



例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

    1
   / \
  2   2
 / \ / \
3  4 4  3


但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    1
   / \
  2   2
   \   \
   3    3


进阶：

你可以运用递归和迭代两种方法解决这个问题吗？
**/

func IsSymmetric(root *TreeNode) bool {

	//就是比较两边右-左是否相同

	var sub func(left *TreeNode, right *TreeNode) bool

	sub = func(left, right *TreeNode) bool {

		//继续比较:左右都存在且相等的情况
		if left != nil && left.Val == right.Val {

			//比较左边的左子树 和右边的右子树
			return sub(left.Left, right.Right) && sub(left.Right, right.Left)
		}

		//有一个为空的情况
		if (left == nil && right != nil) || (left != nil && right == nil) {

			return false
		}
		return true
	}

	return sub(root.Left, root.Right)

}

/*
qn:104. 二叉树的最大深度
给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。
*/

func MaxDepth(root *TreeNode) int {

	var sub func(root *TreeNode, max int) int

	sub = func(root *TreeNode, max int) int {

		if root == nil {

			return max
		} else {
			fmt.Println(root.Val)
			max++
			//获取左有子树高度
			leftDepth := sub(root.Left, max)
			rightDepth := sub(root.Right, max)
			if leftDepth > rightDepth {
				return leftDepth
			} else {
				return rightDepth
			}
		}

	}

	init := 0
	res := sub(root, init)

	return res
}

/**
111. 二叉树的最小深度
给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明：叶子节点是指没有子节点的节点。

示例 1：


输入：root = [3,9,20,null,null,15,7]
输出：2
示例 2：

输入：root = [2,null,3,null,4,null,5,null,6]
输出：5
**/
func MinDepth(root *TreeNode) int {

	var sub func(node *TreeNode, max int) int

	sub = func(node *TreeNode, max int) int {

		if node == nil {

			return max
		} else {
			fmt.Println(node.Val)
			max++

			//有左右节点取小值
			if node.Left != nil && node.Right != nil {

				leftDepth := sub(node.Left, max)
				rightDepth := sub(node.Right, max)

				if leftDepth < rightDepth {
					return leftDepth
				} else {
					return rightDepth
				}

			}
			//只有左节点
			if node.Left != nil && node.Right == nil {
				return sub(node.Left, max)
			}

			//只有右节点
			if node.Right != nil && node.Left == nil {
				return sub(node.Right, max)
			}
			//叶子节点
			return max
		}
	}

	return sub(root, 0)
}

/**
222. 完全二叉树的节点个数
给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。

完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。

输入：root = [1,2,3,4,5,6]
输出：6
示例 2：

输入：root = []
输出：0
示例 3：

输入：root = [1]
输出：1


提示：

树中节点的数目范围是[0, 5 * 104]
0 <= Node.val <= 5 * 104
题目数据保证输入的树是 完全二叉树


进阶：遍历树来统计节点是一种时间复杂度为 O(n) 的简单解决方案。你可以设计一个更快的算法吗？
**/
func CountNodes(root *TreeNode) int {

	res := 0
	//遍历算
	valList, _, _ := root.LevelOrder()
	res = len(valList)

	return res

	//todo 二分查找 + 位运算
}

/**
110. 平衡二叉树
给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
**/

func IsBalanced(root *TreeNode) bool {

	if root.Left == nil || root.Right == nil {
		return false
	}

	leftMax := MaxDepth(root.Left)
	rightMax := MaxDepth(root.Right)

	return math.Abs(float64(leftMax-rightMax)) == 1
}

/*
257. 二叉树的所有路径
给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。

叶子节点 是指没有子节点的节点。
*/

func BinaryTreePaths(root *TreeNode) []string {

	var res [][]int

	var sub func(node *TreeNode, path []int, res *[][]int)

	sub = func(node *TreeNode, path []int, res *[][]int) {

		path = append(path, node.Val)

		//到页子节点
		if node.Left == nil && node.Right == nil {
			fmt.Println("ye", path)
			*res = append(*res, path)
			fmt.Println("res-in:", *res)
			return
		}

		if node.Left != nil {
			sub(node.Left, path, res)

		}
		if node.Right != nil {
			fmt.Println("right", path)
			//todo 一定要全量复制，不然会已经插入的数据会被修改成第二次插入的数据
			newaa := make([]int, len(path))

			copy(newaa, path)
			sub(node.Right, newaa, res)
		}
	}

	sub(root, []int{}, &res)

	fmt.Println("res", res)

	var st []string

	return st
}

/**
404. 左叶子之和
计算给定二叉树的所有左叶子之和。

示例：

    3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
**/
func SumOfLeftLeaves(root *TreeNode) int {

	var sub func(node *TreeNode, sum *int, parent *TreeNode)

	res := 0

	//前序遍历
	sub = func(node *TreeNode, sum *int, parent *TreeNode) {

		if parent != nil && parent.Left == node && node.Left == nil && node.Right == nil {

			*sum = *sum + node.Val
			return
		}
		parent = node
		if node.Left != nil {
			sub(node.Left, sum, parent)
		}
		if node.Right != nil {
			sub(node.Right, sum, parent)
		}

	}

	sub(root, &res, nil)
	return res
}

/*
513. 找树左下角的值
给定一个二叉树的 根节点 root，请找出该二叉树的 最底层 最左边 节点的值。

假设二叉树中至少有一个节点。
*/

func FindBottomLeftValue(root *TreeNode) int {

	//层序遍历最后一层的第一个节点

	if root.Left == nil && root.Right == nil {

		return root.Val
	}

	_, _, valLevelList := root.LevelOrder()

	return valLevelList[len(valLevelList)-1][0]
}

/**
112. 路径总和
给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。

叶子节点 是指没有子节点的节点。
**/

func HasPathSum(root *TreeNode, targetSum int) bool {

	//todo isFind?

	var sub func(node *TreeNode, t int, isFind *bool)

	sub = func(node *TreeNode, t int, isFind *bool) {

		if node == nil {
			return
		}

		remain := t - node.Val

		//叶子节点
		if node.Left == nil && node.Right == nil && remain == 0 {

			*isFind = true
			return

		}

		if node.Left != nil && !(*isFind) {

			sub(node.Left, remain, isFind)
		}

		if node.Right != nil && !(*isFind) {
			sub(node.Right, remain, isFind)
		}
	}

	fmt.Println("----")

	res := false
	sub(root, targetSum, &res)

	return res
}

/**
106. 从中序与后序遍历序列构造二叉树
根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7
**/
func BuildTree(inorder []int, postorder []int) *TreeNode {

	var sub func(l []int, p []int, i *int) *TreeNode

	sub = func(l, p []int, i *int) *TreeNode {

		if *i < 0 {
			return nil
		}

		curVal := p[*i]
		cur := &TreeNode{Val: curVal}
		si := 0

		for k, v := range l {

			if v == curVal {
				si = k
				break
			}
		}

		left := l[0:si]
		right := l[si+1:]
		//todo 为啥不会越界
		//if si <= len(l) -1 {

		//right := l[si+1:]
		//}else {
		//right := make([]int, 0)
		//}

		if len(right) == len(left) && len(right) == 0 {
			return cur
		}
		if len(right) > 0 {

			(*i)--
			cur.Right = sub(right, p, i)
		}
		if len(left) > 0 {
			(*i)--
			cur.Left = sub(left, p, i)
		}

		return cur
	}

	res := len(postorder) - 1

	return sub(inorder, postorder, &res)

}

/**
654. 最大二叉树
给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：

二叉树的根是数组 nums 中的最大元素。
左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
返回有给定数组 nums 构建的 最大二叉树 。



示例 1：


输入：nums = [3,2,1,6,0,5]
输出：[6,3,5,null,2,0,null,null,1]
解释：递归调用如下所示：
- [3,2,1,6,0,5] 中的最大值是 6 ，左边部分是 [3,2,1] ，右边部分是 [0,5] 。
    - [3,2,1] 中的最大值是 3 ，左边部分是 [] ，右边部分是 [2,1] 。
        - 空数组，无子节点。
        - [2,1] 中的最大值是 2 ，左边部分是 [] ，右边部分是 [1] 。
            - 空数组，无子节点。
            - 只有一个元素，所以子节点是一个值为 1 的节点。
    - [0,5] 中的最大值是 5 ，左边部分是 [0] ，右边部分是 [] 。
        - 只有一个元素，所以子节点是一个值为 0 的节点。
        - 空数组，无子节点。
**/

func ConstructMaximumBinaryTree(nums []int) *TreeNode {

	var sub func(n []int) *TreeNode

	sub = func(n []int) *TreeNode {

		if len(n) <= 0 {

			return nil
		}
		maxIndex := 0
		maxValue := n[0]

		for i := 1; i < len(n)-1; i++ {
			if maxValue < n[i] {

				maxIndex = i
				maxValue = n[i]
			}
		}

		curNode := &TreeNode{Val: maxValue}

		curNode.Left = sub(n[:maxIndex])
		curNode.Right = sub(n[maxIndex+1:])

		return curNode
	}

	return sub(nums)
}

/*
617. 合并二叉树
给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

示例 1:

输入:
	Tree 1                     Tree 2
          1                         2
         / \                       / \
        3   2                     1   3
       /                           \   \
      5                             4   7
输出:
合并后的树:
	     3
	    / \
	   4   5
	  / \   \
	 5   4   7
注意: 合并必须从两个树的根节点开始。
*/
func MergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {

	if root1 == nil {
		return root2
	}

	if root2 == nil {

		return root1
	}

	root1.Val += root2.Val

	root1.Left = MergeTrees(root1.Left, root2.Left)
	root1.Right = MergeTrees(root1.Right, root2.Right)

	return root1
}

/**
700. 二叉搜索树中的搜索
给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

例如，

给定二叉搜索树:

        4
       / \
      2   7
     / \
    1   3

和值: 2
你应该返回如下子树:

      2
     / \
    1   3
在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。
**/
func SearchBST(root *TreeNode, val int) *TreeNode {

	if root == nil {
		return nil
	}

	if root.Val == val {

		return root
	}

	if val > root.Val {

		return SearchBST(root.Right, val)
	} else {

		return SearchBST(root.Left, val)
	}
}

/**
98. 验证二叉搜索树
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

有效 二叉搜索树定义如下：

节点的左子树只包含 小于 当前节点的数。
节点的右子树只包含 大于 当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
**/
func IsValidBST(root *TreeNode) bool {

	//中序遍历从小到大
	var sub func(node *TreeNode, last *int, isInit *bool, res *bool)

	sub = func(node *TreeNode, last *int, isInit *bool, res *bool) {

		if node.Left != nil && *res {
			sub(node.Left, last, isInit, res)
		}

		fmt.Println("last", *last, "|cur", node.Val)
		if *isInit == false {
			*last = node.Val
			*isInit = true
		} else {

			if node.Val < *last {

				*res = false
				return
			} else {
				*last = node.Val
			}
		}

		if node.Right != nil && *res {
			sub(node.Right, last, isInit, res)
		}

		return
	}

	isInit, res := false, true
	var last int
	sub(root, &last, &isInit, &res)

	fmt.Println("---")

	return res
}

/**
530. 二叉搜索树的最小绝对差
给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。
**/
func GetMinimumDifference(root *TreeNode) int {

	inOrder, _ := root.InOrderByIteration()

	//占位，懒得判断初始化
	min := 999
	for i := 1; i < len(inOrder); i++ {

		abs := math.Abs(float64(inOrder[i] - inOrder[i-1]))

		if int(abs) < min {
			min = int(abs)
		}

	}

	return min
}

/*

501. 二叉搜索树中的众数
给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。

假定 BST 有如下定义：

结点左子树中所含结点的值小于等于当前结点的值
结点右子树中所含结点的值大于等于当前结点的值
左子树和右子树都是二叉搜索树

*/

func FindMode(root *TreeNode) int {

	var lastMaxCount, lastMaxValue int

	var sub func(node *TreeNode, parent *int, count *int)

	sub = func(node *TreeNode, parent, count *int) {

		if node.Left != nil {

			sub(node.Left, parent, count)
		}

		//第一个元素
		if *count == 0 && *parent == 0 {

			*count++
			*parent = node.Val
			lastMaxCount = *count
			lastMaxValue = node.Val

		} else {

			//当前元素的值等于上一个元素
			if *parent == node.Val {

				*count++
				//上一个元素不等于且个数大于目前最大的
				if *count > lastMaxCount {
					lastMaxCount = *count
					lastMaxValue = *parent
				}

			} else {
				*count = 1
				*parent = node.Val
			}
		}

		if node.Right != nil {

			sub(node.Right, parent, count)
		}
	}

	parent, count := 0, 0

	sub(root, &parent, &count)

	return lastMaxValue
}

/**

236. 二叉树的最近公共祖先
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。


输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。

**/

func LowestCommonAncestor(root, p, q *TreeNode) *TreeNode {

	//todo
	if root == nil {
		return root
	}
	if p == root || q == root {
		return root
	}

	left := LowestCommonAncestor(root.Left, p, q)
	right := LowestCommonAncestor(root.Right, p, q)

	if left != nil && right != nil {

		return root
	}

	if left != nil {
		return left
	}

	if right != nil {

		return right
	}

	return nil
}

/**
235. 二叉搜索树的最近公共祖先
todo
**/

/**
701. 二叉搜索树中的插入操作
给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。

注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。
**/

func InsertIntoBST(root *TreeNode, val int) *TreeNode {

	//不调整结构
	if root == nil {

		return &TreeNode{Val: val}
	}

	if val > root.Val {
		root.Right = InsertIntoBST(root.Right, val)
	} else {

		root.Left = InsertIntoBST(root.Left, val)
	}

	return root
}

/**
450. 删除二叉搜索树中的节点
给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。
说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

示例:

root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。

一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。

    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。

    5
   / \
  2   6
   \   \
    4   7
**/
func DeleteNode(root *TreeNode, key int) *TreeNode {

	if root == nil {
		return nil
	}

	if key == root.Val {

		if root.Left == nil && root.Right == nil {

			return nil
		}

		if root.Left == nil && root.Right != nil {
			return root.Right
		}

		if root.Left != nil && root.Right == nil {
			return root.Left
		}

		if root.Left != nil && root.Right != nil {

			tmp := root.Left

			cur := root.Right

			for cur.Left != nil {

				cur = cur.Left
			}
			cur.Left = tmp

			return root.Right
		}

	} else if key > root.Val {
		root.Right = DeleteNode(root.Right, key)
	} else if key < root.Val {
		root.Left = DeleteNode(root.Left, key)
	}

	return root
	//2.只用左边或者右边,不考虑两边都有。一样的只是就修改了一边
}

/**
669. 修剪二叉搜索树
给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，使得所有节点的值在[low, high]中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系都应当保留）。 可以证明，存在唯一的答案。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。
**/
func TrimBST(root *TreeNode, low int, high int) *TreeNode {

	//前序遍历
	if root == nil {

		return root
	}

	if root.Val < low {
		//todo fix return root.Right
		return TrimBST(root.Right, low, high)
	}

	if root.Val > high {

		return TrimBST(root.Left, low, high)
	}

	root.Left = TrimBST(root.Left, low, high)
	root.Right = TrimBST(root.Right, low, high)

	return root
}

/**
108. 将有序数组转换为二叉搜索树
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
**/
func SortedArrayToBST(nums []int) *TreeNode {

	if len(nums) == 0 {

		return nil
	}

	mid := len(nums) / 2

	left := nums[:mid]
	right := nums[mid+1:]

	cur := &TreeNode{Val: nums[mid]}

	if len(left) > 0 {

		cur.Left = SortedArrayToBST(left)
	}

	if len(right) > 0 {

		cur.Right = SortedArrayToBST(right)
	}
	return cur
}

/**
538. 把二叉搜索树转换为累加树
给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。
注意：本题和 1038: https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree/ 相同

示例 1：

输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
**/

func ConvertBST(root *TreeNode) *TreeNode {

	//右根左 倒叙

	var sub func(node *TreeNode, sum *int)

	sub = func(node *TreeNode, sum *int) {
		if node == nil {
			return
		}

		sub(node.Right, sum)

		node.Val += *sum
		*sum = node.Val

		sub(node.Left, sum)

		return
	}

	s := 0
	sub(root, &s)
	return root
}

/**
77. 组合
给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。

你可以按 任何顺序 返回答案。

1 <= n <= 20
1 <= k <= n

示例 1：

输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
**/

func Combine(n int, k int) [][]int {

	var res [][]int

	var sub func(i int, path []int)

	sub = func(i int, path []int) {

		if len(path) == k {

			tmp := make([]int, k)
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for x := i; x <= n; x++ {

			path = append(path, x)
			sub(x+1, path)
			//fix 123 剔除3回溯
			path = path[:len(path)-1]
			fmt.Println("path:", path, "x", x)
		}
	}

	sub(1, []int{})
	fmt.Println("res", res)
	return res
}

/**
216. 组合总和 III
找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

说明：

所有数字都是正整数。
解集不能包含重复的组合。
示例 1:

输入: k = 3, n = 7
输出: [[1,2,4]]
示例 2:

输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
**/

func CombinationSum3(k int, n int) [][]int {

	var res [][]int

	var sub func(i int, path []int, sum int)

	sub = func(i int, path []int, sum int) {

		length := len(path)
		if length > k || sum > n {

			return
		}

		if length == k && sum == n {

			tmp := make([]int, length)
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for x := i; x < 10; x++ {

			path = append(path, x)
			sub(x+1, path, sum+x)
			path = path[:len(path)-1]

		}

	}

	sub(1, []int{}, 0)

	fmt.Println(res)
	return res
}

/**
17. 电话号码的字母组合
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

示例 1：

输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
**/

/**
**/
func LetterCombinations(digits string) []string {

	var res []string

	digArr := [10]string{
		"",
		"",
		"abc",
		"def",
		"ghi",
		"jkl",
		"mno",
		"pqrs",
		"tuv",
		"wxyz",
	}

	//组合=集合
	fmt.Println(digArr)
	return res
	//todo
}

/**
46. 全排列
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
**/

func Permute(nums []int) [][]int {

	var res [][]int
	var aa func(n []int, i int, path []int)

	aa = func(n []int, i int, path []int) {

		fmt.Printf("i= %v", i)
		fmt.Println("----")
		//	结束条件
		if len(n) == 0 {
			tmp := make([]int, len(path))
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		j := 0
		//各个分支
		for j = 0; j < len(n); j++ {
			fmt.Println("一次循环")
			cur := n[j]
			path = append(path, cur)
			n = append(n[:j], n[j+1:]...)
			fmt.Printf("j=%v i=%v cur=%v n=%v p=%v", j, i, cur, n, path)
			fmt.Println("---")
			aa(n, 0, path)
			fmt.Println("一次回溯")
			//fix//回溯的时候切片也要复原，元素位置不能变
			n = append(n[:j], append([]int{cur}, n[j:]...)...)
			path = path[:len(path)-1]
			fmt.Printf("j=%v i=%v cur=%v n=%v p=%v", j, i, cur, n, path)

			fmt.Println("   ")
		}

		fmt.Println("    ")
		fmt.Printf("j = %v", j)
		fmt.Println("    ")
	}
	aa(nums, 0, []int{})
	fmt.Println(res)
	return res
}

func Permute2(nums []int) [][]int {
	var res [][]int

	var dfs func(n []int, length int, s int, path []int)

	dfs = func(n []int, length, s int, path []int) {

		fmt.Println("    ")
		fmt.Printf("s= %v", s)
		fmt.Println("")

		if len(path) == length {
			tmp := make([]int, length)
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for i := s; i < length; i++ {

			fmt.Println("一次循环")
			n[s], n[i] = n[i], n[s]
			path = append(path, n[s])
			fmt.Printf("n: %v, i: %v, s: %v, p: %v", n, i, s, path)
			dfs(n, length, s+1, path)

			fmt.Println("一次回溯")
			n[s], n[i] = n[i], n[s]
			path = path[:len(path)-1]

			fmt.Printf("n: %v, i: %v, s: %v, p: %v", n, i, s, path)

			fmt.Println("     ")
		}

	}

	dfs(nums, len(nums), 0, []int{})

	fmt.Println(res)
	return res
}

/**
39. 组合总和
给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。

candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。

对于给定的输入，保证和为 target 的唯一组合数少于 150 个。

输入: candidates = [2,3,6,7], target = 7
输出: [[7],[2,2,3]]

输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]

输入: candidates = [2], target = 1
输出: []

输入: candidates = [1], target = 1
输出: [[1]]

输入: candidates = [1], target = 2
输出: [[1,1]]
提示：

1 <= candidates.length <= 30
1 <= candidates[i] <= 200
candidate 中的每个元素都是独一无二的。
1 <= target <= 500

**/
func CombinationSum(candidates []int, target int) [][]int {

	var res [][]int
	var dfs func(num []int, s int, max int, p []int, target int, sum int)

	dfs = func(num []int, s, max int, p []int, target int, sum int) {

		if sum > target {
			return
		}

		if sum == target {
			tmp := make([]int, len(p))
			copy(tmp, p)
			res = append(res, tmp)
			return
		}

		for j := s; j < max; j++ {
			p = append(p, num[j])
			sum += num[j]
			dfs(num, j, max, p, target, sum)
			//fix: 如果我返回了说明找到了一个比目标值大的或者等于的所以当前p得退一个栈
			sum -= p[len(p)-1]
			p = p[:len(p)-1]
		}
	}

	dfs(candidates, 0, len(candidates), []int{}, target, 0)

	fmt.Println(res)
	return res
}

/*

qn: 40. 组合总和 II
给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

注意：解集不能包含重复的组合。



示例 1:

输入: candidates = [10,1,2,7,6,1,5], target = 8,
输出:
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
示例 2:

输入: candidates = [2,5,2,1,2], target = 5,
输出:
[
[1,2,2],
[5]
]


提示:

1 <= candidates.length <= 100
1 <= candidates[i] <= 50
1 <= target <= 30
*/
func CombinationSum2(candidates []int, target int) [][]int {

	var res [][]int

	nums := candidates

	//先排序好去重
	sort.Ints(nums)

	var dfs func(sum int, path []int, start int)

	dfs = func(sum int, path []int, start int) {

		if sum == target {

			for _, v := range res {

				if reflect.DeepEqual(v, path) {
					return
				}
			}
			tmp := make([]int, len(path))
			copy(tmp, path)
			res = append(res, tmp)
			return
		}
		if sum > target {
			return
		}

		for j := start; j < len(nums); j++ {

			//if j > 0 && nums[j] == nums[j-1] {
			//continue
			//}
			path = append(path, nums[j])
			sum += nums[j]
			dfs(sum, path, j+1)

			sum -= nums[j]
			path = path[:len(path)-1]
		}
	}

	dfs(0, []int{}, 0)
	fmt.Println("CombinationSum2", res)

	//2. 递归树横向不能有相同的
	return res
}

/**
131. 分割回文串
给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

回文串 是正着读和反着读都一样的字符串。



示例 1：

输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
示例 2：

输入：s = "a"
输出：[["a"]]
提示：
1 <= s.length <= 16
s 仅由小写英文字母组成
**/
func Partition(s string) [][]string {
	var res [][]string

	var sStrings []string

	for _, v := range s {
		sStrings = append(sStrings, string(v))
	}
	length := len(sStrings)

	fmt.Println(sStrings)

	isPar := func(num []string, start, end int) bool {
		res := true
		left, right := start, end

		for left <= right {
			if num[left] != num[right] {
				res = false
				break
			}
			left++
			right--
		}

		return res
	}

	var dfs func(start int, path []string)

	dfs = func(start int, path []string) {

		if start >= length {

			tmp := make([]string, len(path))
			copy(tmp, path)
			res = append(res, tmp)
			return
		}
		i := start
		for i = start; i < length; i++ {

			//fix 重star开始
			isP := isPar(sStrings, start, i)
			if !isP {
				continue
			}

			part := strings.Join(sStrings[start:i+1], "")
			path = append(path, part)
			fmt.Println("1:path=", path, "start", start, "i=", i)
			dfs(i+1, path)
			path = path[:len(path)-1]

			fmt.Println("2:path=", path, "start", start, "i=", i)
		}
		fmt.Println("3:path=", path, "start", start, "i=", i, "循环结束")
	}

	dfs(0, []string{})

	fmt.Println(res)
	return res
}

/**
93. 复原 IP 地址
给定一个只包含数字的字符串，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。

有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。

输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]

输入：s = "0000"
输出：["0.0.0.0"]

输入：s = "1111"
输出：["1.1.1.1"]

输入：s = "010010"
输出：["0.10.0.10","0.100.1.0"]

输入：s = "101023"
输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]

0 <= s.length <= 3000
s 仅由数字组成
**/
func RestoreIpAddresses(s string) []string {
	var res []string
	fmt.Println(s[1])
	// length := len(s)

	var dfs func(end int, path []string)

	dfs = func(end int, path []string) {
		//todo 写不动了
	}
	dfs(0, []string{})
	return res
}

/**
78. 子集
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

输入：nums = [0]
输出：[[],[0]]

1 <= nums.length <= 10
-10 <= nums[i] <= 10
nums 中的所有元素 互不相同
**/
func Subsets(nums []int) [][]int {
	var res [][]int
	res = append(res, []int{})
	length := len(nums)

	var dfs func(start int, path []int)

	dfs = func(start int, path []int) {

		if start >= length {
			return
		}

		for i := start; i < length; i++ {

			path = append(path, nums[i])
			tmp := make([]int, len(path))
			copy(tmp, path)
			res = append(res, tmp)
			dfs(i+1, path)
			path = path[:len(path)-1]
			fmt.Println("back", "path:", path, "i=:", i)
		}
	}
	dfs(0, []int{})
	fmt.Println("res:", res)
	return res
}

/*
90. 子集 II
给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。



示例 1：

输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
示例 2：

输入：nums = [0]
输出：[[],[0]]
*/
func SubsetsWithDup(nums []int) [][]int {

	var res [][]int
	res = append(res, []int{})
	length := len(nums)

	var dfs func(start int, path []int)

	dfs = func(start int, path []int) {

		for i := start; i < length; i++ {

			//fix这个就很吊了
			if i > start && nums[i] == nums[i-1] {
				continue
			}
			path = append(path, nums[i])
			tmp := make([]int, len(path))
			copy(tmp, path)
			res = append(res, tmp)
			dfs(i+1, path)
			path = path[:len(path)-1]
			fmt.Println("back", "i=:", i, "path=", path)
		}
	}
	dfs(0, []int{})
	fmt.Println("res:", res)
	return res
}

/**
491. 递增子序列
给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素 。你可以按 任意顺序 返回答案。

数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。

输入：nums = [4,6,7,7]
输出：[[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]

输入：nums = [4,4,3,2,1]
输出：[[4,4]]

1 <= nums.length <= 15
-100 <= nums[i] <= 100
**/
func FindSubsequences(nums []int) [][]int {
	var res [][]int

	length := len(nums)

	var dfs func(start int, path []int)
	dfs = func(start int, path []int) {

		if start == length {

			return
		}

		//记录当前层出现的记录
		history := make(map[int]bool)
		//每次递归都是找一个比当前大的元素
		for i := start; i < length; i++ {

			current := nums[i]
			//当前元素小于最后一个元素
			if len(path) > 0 && path[len(path)-1] > current {
				continue
			}
			//TODO 当前元素已经在本层出现过
			if history[current] == true {
				continue
			}

			path = append(path, current)
			history[current] = true
			if len(path) > 1 {

				tmp := make([]int, len(path))
				copy(tmp, path)
				res = append(res, tmp)
			}
			dfs(i+1, path)
			path = path[:len(path)-1]
		}
	}
	dfs(0, []int{})
	fmt.Println(res)
	return res
}

/**
47. 全排列 II
给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。

输入：nums = [1,1,2]
[[1,1,2],
 [1,2,1],
 [2,1,1]]

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

提示：

1 <= nums.length <= 8
-10 <= nums[i] <= 10
**/
func PermuteUnique(nums []int) [][]int {
	var res [][]int

	var dfs func(n []int, path []int)
	dfs = func(n []int, path []int) {

		if len(n) == 0 {

			tmp := make([]int, len(path))
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		history := make(map[int]bool)
		for i := 0; i < len(n); i++ {
			cur := n[i]
			if history[cur] == true {
				continue
			}
			history[cur] = true
			path = append(path, cur)
			n = append(n[:i], n[i+1:]...)
			dfs(n, path)
			n = append(n[:i], append([]int{cur}, n[i:]...)...)
			path = path[:len(path)-1]
		}
	}
	dfs(nums, []int{})
	fmt.Println(res)
	return res
}

/**
332. 重新安排行程
给你一份航线列表 tickets ，其中 tickets[i] = [fromi, toi] 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。

所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。如果存在多种有效的行程，请你按字典排序返回最小的行程组合。

例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前。
假定所有机票至少存在一种合理的行程。且所有的机票 必须都用一次 且 只能用一次。


输入：tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
输出：["JFK","MUC","LHR","SFO","SJC"

输入：tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
输出：["JFK","ATL","JFK","SFO","ATL","SFO"]
解释：另一种有效的行程是 ["JFK","SFO","ATL","JFK","ATL","SFO"] ，但是它字典排序更大更靠后。
**/
func FindItinerary(tickets [][]string) []string {
	var res []string
	//TODO
	return res
}

/**
51. N 皇后
n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
**/
func SolveNQueens(n int) [][]string {
	var res [][]string

	//init
	var resSlice [][]string
	for i := 0; i < n; i++ {
		var tmp []string
		for j := 0; j < n; j++ {

			tmp = append(tmp, ".")
		}
		resSlice = append(resSlice, tmp)
	}

	canPlace := func(x int, y int, r [][]string) bool {

		//同一x 上是否有元素
		for t := 0; t < x; t++ {

			if (r)[t][y] == "Q" {
				return false
			}
		}

		//左上角是否有元素
		tx := x - 1
		ty := y - 1

		for tx >= 0 && ty >= 0 {

			if (r)[tx][ty] == "Q" {
				return false
			}

			tx--
			ty--
		}

		//右上角是否有元素
		tx = x - 1
		ty = y + 1
		for tx >= 0 && ty < n {

			if (r)[tx][ty] == "Q" {
				return false
			}

			tx--
			ty++
		}

		return true
	}
	//TODO 左下角 x+y互斥 右下角x-y互斥  但是有互斥的结果并不代表不能插入

	var dfs func(x int, r [][]string)

	dfs = func(x int, r [][]string) {

		if x == n {

			tmp := make([]string, n)
			for k := 0; k < n; k++ {
				tmp[k] = strings.Join(r[k], "")
			}

			res = append(res, tmp)
			return
		}

		for y := 0; y < n; y++ {

			if canPlace(x, y, r) == false {
				continue
			}

			resSlice[x][y] = "Q"

			dfs(x+1, r)

			resSlice[x][y] = "."
		}
	}
	dfs(0, resSlice)
	fmt.Println(res)
	return res
}

/**
37. 解数独
编写一个程序，通过填充空格来解决数独问题。

数独的解法需 遵循如下规则：

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用 '.' 表示。

输入：board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
输出：[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
**/

func SolveSudoku(board [][]byte)  {

    //当找到第n个未填充数字的格子，此时尝试在里面填写1可以保证当前行不冲突，列不冲突，小方块不冲突，那么就在第n个格子里面填写1，
    //然后查看第n+1个格子，如果第n+1个格子发现填写1-9都冲突，
    //说明第n个格子填写1的情况下无法找到解，此时我们需要回到第n个格子，填写另外一个不冲突的数字。一直到能填充到第81个格子为止。
    canPlace := func (board [][]byte, x int, y int, target byte) bool{
       

        //竖方向是否有形同
        for j := 0; j< 9; j++ {
            
            if board[j][y] == target {
                
                return false
            }
        } 


        //横方向是否有相同
        for j:= 0; j<9; j++ {
            
            if board[x][j] == target {

                return false
            }
        } 

        //九宫格是否有相同
        for ix := (x/3)*3; ix <= (x/3)*3 +2; ix ++ {

            for iy := (y/3)*3; iy <= (y/3)*3 +2; iy ++ {

                if board[ix][iy] == target {
                    return false
                }
            }

        }

        return true
    }

    var dfs func(board [][]byte) bool

    dfs = func(board [][]byte) bool {

        for x:= 0; x< 9; x++ {

            for y:=0; y<9; y++ {

                if board[x][y] != '.' {
                    continue
                }

                //填充：fix j数据类型为byte
                for j := '1'; j<='9'; j++ {

                    if canPlace(board, x, y, byte(j)) == false {
                        continue
                    }

                    //找到一个可以填充的元素
                    board[x][y] = byte(j)
                    res := dfs(board)
                    if res == true {
                        return true
                    }

                    board[x][y] = '.' 
                }

                return false

            }
        }

        return true
    }
        
    dfs(board)

    fmt.Println(board)
    fmt.Printf("%c", board)
}

/**
455. 分发饼干
假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

 
示例 1:

输入: g = [1,2,3], s = [1,1]
输出: 1
解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。
**/
func FindContentChildren(g []int, s []int) int {
    //s 是饼干 g是小孩

    res := 0
    
    sort.Ints(g)
    sort.Ints(s)
    
    gk, sk := 0, 0

    for gk < len(g) && sk < len(s){

        if s[sk] >= g[gk] {
            gk++
            sk++
            res++
        }else{

            sk++
        }
    }

    return res
}

/**

376. 摆动序列
如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。

相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。

输入：nums = [1,7,4,9,2,5]
输出：6
解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。

输入：nums = [1,17,5,10,13,15,10,5,16,8]
输出：7
解释：这个序列包含几个长度为 7 摆动序列。
其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。

输入：nums = [1,2,3,4,5,6,7,8,9]
输出：2
**/
func WiggleMaxLength(nums []int) int {
   
    res := 1
   
    pre := 0
    var sub []int
    
   //只到倒数第二个元素 
    for i:= 0; i < len(nums) -1; i++ {

        //当前结果
        cur := nums[i+1] - nums[i]

        if (cur > 0 && pre <= 0)  || (cur < 0 && pre >= 0){
            res ++
            sub = append(sub, nums[i])
            pre = cur
        }
    }

    //fix 最后一个先算进去了。。。。最后一个元素得加上去
    sub = append(sub,nums[len(nums)-1])
    fmt.Println(sub)
    return res
}

/**
53. 最大子序和
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
**/
func MaxSubArray(nums []int) int {

    res := 0

    //暴力破解
    for i:= 0; i< len(nums); i++ {
       
        sum := 0

        for j := i; j< len(nums); j++ {

            sum += nums[j]
            fmt.Println("sum", sum)
            if sum > res {
                res = sum
            }
        }
        fmt.Println("---res", res)
    }


    fmt.Println(res)
    return res
}

func MaxSubArray2(nums []int) int {

    res := 0

    //如果前和再加上当前的数小于0那么子序列结束
    max := 0
    for i:=0;i<len(nums); i++ {

        if max+nums[i] <= 0 {
            max = 0
            continue
        }else {

            max += nums[i]
            if max > res {
                res = max
            }
        }
    }
    //test中文
    //架空了极乐空间了空降

    fmt.Println(res)
    return res
}

/*
122. 买卖股票的最佳时机 II
给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

 

示例 1:

输入: prices = [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
*/
func MaxProfit(prices []int) int {
    
    //误区. 同一天可以先卖再买所以可以先求差再取整:所以暴力没意义啊

    res := 0
    
    for i := 1; i< len(prices); i++ {

        cur := prices[i] - prices[i-1]

        if cur > 0 {
            res += cur
        }
    }

    return res
}

/*
55. 跳跃游戏
给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。

输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
*/
func CanJump(nums []int) bool {
    
    max := 0

    for i:= 0 ;i<len(nums);i++ {

        //找到一个点等于0，且前面点能跳的最大的都不超过这个点则到不了最后一个
        if nums[i] == 0 && max <= i {
            return false
        }

        tMax := i+nums[i]

        if tMax >= max {
            max = tMax
        }
    }

    return true 
}


/**
45. 跳跃游戏 II
给你一个非负整数数组 nums ，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
示例 2:

输入: nums = [2,3,0,1,4]
输出: 2

1 <= nums.length <= 104
0 <= nums[i] <= 1000


TODO 在不得不跳的时候再跳

cur_max表示跳i步能达到最远的坐标
next_max表示在cur_max之前的位置跳一步能达到下一个最远的位置
例如：
[4,2,7,3,1,1,3,1,1,1]
第一步从4开始，最远跳到数字1的位置，cur_max = 0+4=4
那跳两步最远的位置为max(1+2, 2+7, 3+3, 4+1)中的一个，
其中的加法操作是i+nums[i]。不难得出next_max=2+7=9

遍历数组，当i==cur_max，step+1，
因为上一次跳跃最远能到cur_max，要超过cur_max必须再跳一次，而再跳一次能达到的最远距离为next_max
**/
func Jump(nums []int) int {
   

    fmt.Println("---")

    res := 0

    for i:= 0 ; i< len(nums); {
       
        //每次进来走一步
        res ++

        max := i+nums[i]

        nextMax := 0
        for j:= i+1; j <= max; j++ {

            if j+nums[j] > nextMax {
                nextMax = j+ nums[j]
            }
        }

        res ++

        fmt.Println("nextMax: ", nextMax, "max", max)

        if nextMax >= len(nums) -1 {
            return res
        }


        //步进跳到nextMax
        i = nextMax

    }

    return res
}


/**
1005. K 次取反后最大化的数组和
给定一个整数数组 A，我们只能用以下方法修改该数组：我们选择某个索引 i 并将 A[i] 替换为 -A[i]，然后总共重复这个过程 K 次。（我们可以多次选择同一个索引 i。）

以这种方式修改数组后，返回数组可能的最大和。

 

示例 1：

输入：A = [4,2,3], K = 1
输出：5
解释：选择索引 (1,) ，然后 A 变为 [4,-2,3]。
示例 2：

输入：A = [3,-1,0,2], K = 3
输出：6
解释：选择索引 (1, 2, 2) ，然后 A 变为 [3,1,0,2]。
示例 3：

输入：A = [2,-3,-1,5,-4], K = 2
输出：13
解释：选择索引 (1, 4) ，然后 A 变为 [2,3,-1,5,4]。
 

提示：

1 <= A.length <= 10000
1 <= K <= 10000
-100 <= A[i] <= 100
**/

func LargestSumAfterKNegations(nums []int, k int) int {

    //TODO  如果先排序那么替换以后顺序就不是有序的了，还得重排序才能是有序的所以。按照绝对值排序
    sort.Slice(nums, func(i, j int) bool {
        return math.Abs(float64(nums[i])) > math.Abs(float64(nums[j]))
    })

    sum := 0

    //遍历如遇到负数转正数
    for i:= 0; i < len(nums); i++ {

        if k > 0 && nums[i] < 0 {

            nums[i] *= -1
            k--
        }
        sum += nums[i]
    }

    if k == 0 {
        return sum
    }

    //将最后一个数一直转且k是奇数
    if k > 0  && k%2 != 0{

        sum = sum - nums[len(nums)-1] + nums[len(nums)-1] * -1
    }
    fmt.Println(nums)
    return sum
}

/**
134. 加油站
在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明: 

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。
示例 1:

输入: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

输出: 3

解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。
示例 2:

输入: 
gas  = [2,3,4]
cost = [3,4,3]

输出: -1

解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。


因此，无论怎样，你都不可能绕环路行驶一周。
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]
		-2,-2,-2,3,3

        1,2,4,3,5
		3,4,1,5,2
		-2,-2,3,-1,3
**/

func CanCompleteCircuit(gas []int, cost []int) int {


    //如果总和小于0则走不出去

    remain := 0
    cur := 0
    start := 0
    for i:= 0; i< len(gas); i++ {

        remain += gas[i]- cost[i]
        cur += gas[i]-cost[i]

        if cur < 0 {
            start = i+1
            cur = 0
        }
    }

    if remain < 0 {
        return -1
    }
    
    return start

}
