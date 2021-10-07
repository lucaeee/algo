package algo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGeneralLinkList(t *testing.T) {

	testSlice := []int{1, 2, 3, 4}

	res := GeneralLinkList(testSlice)
	assert.Equal(t, 1, res.Val)
	assert.Equal(t, 2, res.Next.Val)
	assert.Equal(t, 3, res.Next.Next.Val)
	assert.Equal(t, 4, res.Next.Next.Next.Val)
}

func TestGetListValues(t *testing.T) {

	testSlice := []int{1, 2, 3, 4}

	res := GeneralLinkList(testSlice)
	values := res.getListValues()

	assert.Equal(t, []int{1, 2, 3, 4}, values)
}

func TestStack(t *testing.T) {
	stack := Stack{}
	stack.Push(5)
	stack.Push(4)
	stack.Push(3)
	stack.Push(2)
	stack.Push(1)

	assert.Equal(t, 1, stack.Pop())
	assert.Equal(t, 2, stack.Pop())
	assert.Equal(t, 3, stack.Pop())
	assert.Equal(t, 4, stack.Pop())
	assert.Equal(t, 5, stack.Pop())

	assert.Equal(t, false, stack.IsEmpty())
}

func TestGeneralTree(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5, 6}
	root := GeneralTree(testSlice)

    assert.Equal(t, 2, root.Left.Val)
    assert.Equal(t, 4, root.Left.Left.Val)
    assert.Equal(t, 5, root.Left.Right.Val)

    assert.Equal(t, 3, root.Right.Val)
    assert.Equal(t, 6, root.Right.Left.Val)

}
func TestLevelOrder(t *testing.T) {
    testSlice := []int{7,6,5, 3, 4, 9,  8}
	root := GeneralTree(testSlice)

    res, _,_ := root.LevelOrder()
    assert.Equal(t, []int{7,6,5, 3, 4, 9,  8}, res)
}

func TestPrevOrderByRecursive(t *testing.T) {
    testSlice := []int{1, 2, 3, 4, 5, 6}
	root := GeneralTree(testSlice)

    res, _ := root.PrevOrderByRecursive()
    assert.Equal(t, []int{1,2,4,5,3,6}, res)
}

func TestPrevOrderByIteration(t *testing.T) {
    testSlice := []int{7,6,5, 3, 4, -999, -999, 9, -999, -999, 8}
	root := GeneralTree(testSlice)

    res, _ := root.PrevOrderByIteration()
    assert.Equal(t, []int{7,6,3,9,4,8,5}, res)
}

func TestInOrderByRecursive(t *testing.T) {
    testSlice := []int{7,6,5, 3, 4, -999, -999, 9, -999, -999, 8}
	root := GeneralTree(testSlice)

    res, _ := root.InOrderByRecursive()
    assert.Equal(t, []int{9,3,6,4,8,7,5}, res)
}

func TestInOrderByIteration(t *testing.T) {
    testSlice := []int{7,6,5, 3, 4, -999, -999, 9, -999, -999, 8}
	root := GeneralTree(testSlice)

    res, _ := root.InOrderByIteration()
    assert.Equal(t, []int{9,3,6,4,8,7,5}, res)
}


func TestNextOrderByRecursive(t *testing.T) {
    testSlice := []int{7,6,5, 3, 4, -999, -999, 9, -999, -999, 8}
	root := GeneralTree(testSlice)

    res, _ := root.NextOrderByRecursive()
    assert.Equal(t, []int{9,3,8,4,6,5,7}, res)
}

func TestNextOrderByIteration(t *testing.T) {
    testSlice := []int{7,6,5, 3, 4, -999, -999, 9, -999, -999, 8}
	root := GeneralTree(testSlice)

    res, _ := root.NextOrderByIteration()
    assert.Equal(t, []int{9,3,8,4,6,5,7}, res)
}
