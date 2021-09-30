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


func TestStack (t *testing.T) {
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
