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
