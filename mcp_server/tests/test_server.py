"""End-to-end tests through a real MCP client connected in-process to the server."""

from mcp import Client


def _data(result):
    assert not result.is_error, result.content
    return result.structured_content


async def test_lists_all_tools(server):
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {"search_otc_labels", "get_label", "normalize_drug", "list_patients",
                     "get_patient_profile", "check_contraindications"}


async def test_full_flow_patient_search_check(server):
    async with Client(server) as client:
        patients = _data(await client.call_tool("list_patients", {}))["result"]
        assert {p["patient_id"] for p in patients} >= {"demo-older-warfarin", "demo-pregnant"}

        profile = _data(await client.call_tool(
            "get_patient_profile", {"patient_id": "demo-older-warfarin"}))
        assert profile["age_years"] >= 68

        found = _data(await client.call_tool(
            "search_otc_labels", {"symptoms": "headache and muscle aches", "top_k": 3}))
        set_ids = [h["set_id"] for h in found["hits"]]
        assert "test-ibuprofen-0001" in set_ids

        report = _data(await client.call_tool("check_contraindications", {
            "set_id": "test-ibuprofen-0001", "patient_id": "demo-older-warfarin"}))
        assert report["verdict"] == "caution"
        assert report["source_url"].startswith("https://dailymed.nlm.nih.gov/")


async def test_get_label_unknown_set_id(server):
    async with Client(server) as client:
        result = _data(await client.call_tool("get_label", {"set_id": "does-not-exist"}))
    assert result["found"] is False


async def test_bad_patient_id_returns_tool_error(server):
    async with Client(server) as client:
        result = await client.call_tool("get_patient_profile", {"patient_id": "../secret"})
    assert result.is_error
