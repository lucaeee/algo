package algo

import "fmt"

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
