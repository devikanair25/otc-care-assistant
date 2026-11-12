import httpx
import respx

from otc_mcp.clients.fhir import FHIRServerSource
from otc_mcp.clients.openfda import BASE_URL as OPENFDA
from otc_mcp.clients.openfda import OpenFDAClient
from otc_mcp.clients.rxnav import BASE_URL as RXNAV
from otc_mcp.clients.rxnav import RxNavClient


@respx.mock
def test_rxnav_brand_name_to_ingredient():
    respx.get(f"{RXNAV}/rxcui.json").mock(
        return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["153010"]}}))
    respx.get(f"{RXNAV}/rxcui/153010/properties.json").mock(
        return_value=httpx.Response(200, json={"properties": {"name": "Advil", "tty": "BN"}}))
    respx.get(f"{RXNAV}/rxcui/153010/related.json").mock(return_value=httpx.Response(
        200, json={"relatedGroup": {"conceptGroup": [
            {"tty": "IN", "conceptProperties": [{"rxcui": "5640", "name": "ibuprofen"}]}]}}))
    concept = RxNavClient().normalize("Advil")
    assert concept.found and concept.ingredients == ["ibuprofen"] and concept.term_type == "BN"


@respx.mock
def test_rxnav_falls_back_to_approximate_match():
    respx.get(f"{RXNAV}/rxcui.json").mock(
        return_value=httpx.Response(200, json={"idGroup": {}}))
    respx.get(f"{RXNAV}/approximateTerm.json").mock(return_value=httpx.Response(
        200, json={"approximateGroup": {"candidate": [{"rxcui": "161"}]}}))
    respx.get(f"{RXNAV}/rxcui/161/properties.json").mock(return_value=httpx.Response(
        200, json={"properties": {"name": "acetaminophen", "tty": "IN"}}))
    concept = RxNavClient().normalize("acetaminofen")
    assert concept.found and concept.ingredients == ["acetaminophen"]


@respx.mock
def test_rxnav_network_error_returns_not_found():
    respx.get(f"{RXNAV}/rxcui.json").mock(side_effect=httpx.ConnectError("down"))
    concept = RxNavClient().normalize("Advil")
    assert not concept.found and "failed" in concept.message


@respx.mock
def test_openfda_label_lookup(raw_records):
    route = respx.get(OPENFDA).mock(
        return_value=httpx.Response(200, json={"results": [raw_records[2]]}))
    label = OpenFDAClient().label_by_set_id("test-loratadine-0001")
    assert label.generic_name == "loratadine"
    assert route.calls.last.request.url.params["search"] == 'set_id:"test-loratadine-0001"'


@respx.mock
def test_openfda_404_means_no_results():
    respx.get(OPENFDA).mock(return_value=httpx.Response(404, json={"error": {}}))
    assert OpenFDAClient().otc_labels_for_ingredient("unobtainium") == []


@respx.mock
def test_fhir_server_follows_paging():
    base = "https://fhir.example.org/fhir"
    page2 = f"{base}/page2"
    respx.get(f"{base}/Patient/p1/$everything").mock(return_value=httpx.Response(200, json={
        "entry": [{"resource": {"resourceType": "Patient", "id": "p1"}}],
        "link": [{"relation": "next", "url": page2}]}))
    respx.get(page2).mock(return_value=httpx.Response(200, json={
        "entry": [{"resource": {"resourceType": "Condition", "id": "c1"}}], "link": []}))
    bundle = FHIRServerSource(base).get_bundle("p1")
    assert [e["resource"]["resourceType"] for e in bundle["entry"]] == ["Patient", "Condition"]
