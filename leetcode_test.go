package algo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBinarySearch(t *testing.T) {

	testSlice := []int{0, 1, 3, 4, 5, 6}
	res := BinarySearch(testSlice, 2)
	assert.Equal(t, false, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 0)
	assert.Equal(t, true, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 6)
	assert.Equal(t, true, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, -1)
	assert.Equal(t, false, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 7)
	assert.Equal(t, false, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 5)
	assert.Equal(t, true, res)
}

func TestRemoveElement(t *testing.T) {

	testSlice := []int{1, 1, 2, 3, 1, 3}
	res := removeElement(testSlice, 1)
	assert.Equal(t, 3, res)
	res = removeElement(testSlice, 3)
	assert.Equal(t, 4, res)
	res = removeElement(testSlice, 2)
	assert.Equal(t, 5, res)

	testSlice = []int{1}
	res = removeElement(testSlice, 1)
	assert.Equal(t, 0, res)
	res = removeElement(testSlice, 2)
	assert.Equal(t, 1, res)
}

func TestRemoveElement2(t *testing.T) {

	testSlice := []int{1, 1, 2, 3, 1, 3}
	res := removeElement2(testSlice, 1)
	assert.Equal(t, 3, res)

	testSlice = []int{1, 1, 2, 3, 1, 3}
	res = removeElement2(testSlice, 3)
	assert.Equal(t, 4, res)

	testSlice = []int{1, 1, 2, 3, 1, 3}
	res = removeElement2(testSlice, 2)
	assert.Equal(t, 5, res)
}

func TestSortedSquares(t *testing.T) {

	testSlice := []int{-4, -1, 0, 3, 10}
	res := sortedSquares(testSlice)
	assert.Equal(t, []int{0, 1, 9, 16, 100}, res)
}

func TestSortedSquares2(t *testing.T) {

	testSlice := []int{-4, -1, 0, 3, 10}
	res := sortedSquares2(testSlice)
	assert.Equal(t, []int{0, 1, 9, 16, 100}, res)
}

func TestMinSubArrayLen(t *testing.T) {

	testSlice := []int{2, 3, 1, 2, 4, 3}
	res := minSubArrayLen(7, testSlice)
	assert.Equal(t, 2, res)

	res = minSubArrayLen(100, testSlice)
	assert.Equal(t, 0, res)

	res = minSubArrayLen(1, testSlice)
	assert.Equal(t, 1, res)
}

func TestMinSubArrayLen2(t *testing.T) {

	testSlice := []int{2, 3, 1, 2, 4, 3}
	res := minSubArrayLen2(7, testSlice)
	assert.Equal(t, 2, res)

	res = minSubArrayLen2(100, testSlice)
	assert.Equal(t, 0, res)

	res = minSubArrayLen2(1, testSlice)
	assert.Equal(t, 1, res)
}

func TestGenerateMatrix(t *testing.T) {

	res := generateMatrix(5)

	assert.EqualValues(t, [][]int{{1, 2, 3, 4, 5}, {16, 17, 18, 19, 6}, {15, 24, 25, 20, 7}, {14, 23, 22, 21, 8}, {13, 12, 11, 10, 9}}, res)
}

func TestRemoveElements(t *testing.T) {

	testSlice := []int{1, 2, 6, 3, 4, 5, 6}
	head := GeneralLinkList(testSlice)
	res := removeElements(head, 1)
	assert.Equal(t, 2, res.Val)
	assert.Equal(t, 6, res.Next.Val)

	testSlice = []int{1, 2, 6, 3, 4, 5, 6}
	head = GeneralLinkList(testSlice)
	res = removeElements(head, 2)
	assert.Equal(t, 1, res.Val)
	assert.Equal(t, 6, res.Next.Val)

}

func TestReverseList(t *testing.T) {

	testSlice := []int{1, 2, 3, 4}

	res := GeneralLinkList(testSlice)
	newHead := reverseList(res)
	assert.Equal(t, 4, newHead.Val)
	assert.Equal(t, 3, newHead.Next.Val)
	assert.Equal(t, 2, newHead.Next.Next.Val)
	assert.Equal(t, 1, newHead.Next.Next.Next.Val)
}

func TestSwapPairs(t *testing.T) {

	testSlice := []int{1, 2, 3, 4}
	res := GeneralLinkList(testSlice)
	swap := swapPairs(res)
	t.Logf("swap %v", swap)
	swapSlice := swap.getListValues()
	assert.Equal(t, []int{2, 1, 4, 3}, swapSlice)

}

func TestRemoveNthFromEnd(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5}
	head := GeneralLinkList(testSlice)
	after := removeNthFromEnd(head, 2)
	assert.Equal(t, []int{1, 2, 3, 5}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd(head, 1)
	assert.Equal(t, []int{1, 2, 3, 4}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd(head, 5)
	assert.Equal(t, []int{2, 3, 4, 5}, after.getListValues())

}

func TestRemoveNthFromEnd2(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5}
	head := GeneralLinkList(testSlice)
	after := removeNthFromEnd2(head, 2)
	assert.Equal(t, []int{1, 2, 3, 5}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd2(head, 1)
	assert.Equal(t, []int{1, 2, 3, 4}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd2(head, 5)
	assert.Equal(t, []int{2, 3, 4, 5}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd2(head, 6)
	assert.Equal(t, []int{1, 2, 3, 4, 5}, after.getListValues())

}

func TestGetIntersectionNode(t *testing.T) {

	slice1 := []int{1, 2, 3, 4, 5}
	head1 := GeneralLinkList(slice1)

	slice2 := []int{7, 8}
	head2 := GeneralLinkList(slice2)

	head2.Next.Next = head1.Next.Next //3
	assert.Equal(t, head2.Next.Next, GetIntersectionNode(head1, head2))
}

func TestDetectCycle(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5}
	head := GeneralLinkList(testSlice)
	head.Next.Next.Next.Next.Next = head
	cycle := DetectCycle(head)
	assert.Equal(t, head, cycle)

}

func TestIsAnagram(t *testing.T) {

	res := IsAnagram("anagram", "nagaram")
	assert.Equal(t, true, res)

	res = IsAnagram("rat", "car")
	assert.Equal(t, false, res)
}

func TestIntersection(t *testing.T) {

	res := Intersection([]int{1, 2, 2, 1}, []int{2, 2})
	assert.Equal(t, []int{2}, res)

	res = Intersection([]int{4, 9, 5}, []int{9, 4, 9, 8, 4})
	assert.Equal(t, []int{9, 4}, res)

}

func TestIsHappy(t *testing.T) {

	// res := IsHappy(19)
	// assert.Equal(t, true, res)

	res := IsHappy(2)
	assert.Equal(t, false, res)
}

func TestTwoSum(t *testing.T) {

	testSlice := []int{2, 7, 11, 15}

	assert.Equal(t, []int{0, 1}, TwoSum(testSlice, 9))
}

func TestFourSumCount(t *testing.T) {

	assert.Equal(t, 2, FourSumCount([]int{1, 2}, []int{-2, -1}, []int{-1, 2}, []int{0, 2}))
}

func TestCanConstruct(t *testing.T) {

	assert.Equal(t, false, CanConstruct("a", "b"))
	assert.Equal(t, false, CanConstruct("aa", "ab"))
	assert.Equal(t, true, CanConstruct("aa", "aab"))
}

func TestThreeSum(t *testing.T) {
	_ = ThreeSum([]int{-1, 0, 1, 2, -1, -4})
}

func TestFourSum(t *testing.T) {

	_ = FourSum([]int{1, 0, -1, 0, -2, 2}, 0)
}
