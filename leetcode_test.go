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
