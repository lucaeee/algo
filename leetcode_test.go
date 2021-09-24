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
