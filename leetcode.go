package algo

import (
	"bytes"
	"encoding/json"
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
