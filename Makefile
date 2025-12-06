.PHONY: release

# Usage: make release v=0.1.1
release:
	@if [ -z "$(v)" ]; then echo "Usage: make release v=0.1.1"; exit 1; fi
	@echo "Releasing version $(v)..."
	sed -i '' 's/^version = ".*"/version = "$(v)"/' pyproject.toml
	git add pyproject.toml
	git commit -m "Bump version to $(v)"
	git push origin main
	git tag v$(v)
	git push origin v$(v)
	@echo "Released v$(v)"
