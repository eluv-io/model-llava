IMAGE_NAME := llava

include buildscripts/Makefile

test: build
	@mkdir -p $(shell pwd)/.cache
	@rm -f tags/1.mp4_tags.json tags/2.mp4_tags.json
	@test -f test/1.mp4 || (echo "Error: test/1.mp4 not found" && exit 1)
	@test -f test/2.mp4 || (echo "Error: test/2.mp4 not found" && exit 1)
	$(CONTAINER_SYSTEM) run  \
		--volume=$(shell pwd)/test:/elv/test:ro \
		--volume=$(shell pwd)/tags:/elv/tags \
		--volume=$(shell pwd)/.cache:/root/.cache \
		--network host $(IMAGE_NAME):$(IMAGE_TAG) test/1.mp4 test/2.mp4
	@test -f tags/1.mp4_tags.json || (echo "Error: tags/1.mp4_tags.json not found" && exit 1)
	@test -f tags/2.mp4_tags.json || (echo "Error: tags/2.mp4_tags.json not found" && exit 1)
	@echo "TESTS PASSED! 🎉"