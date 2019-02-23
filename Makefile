test:
	cd scripts && docker-compose run test
.PHONY: test

benchmark:
	cd scripts && docker-compose run benchmark
.PHONY: benchmark