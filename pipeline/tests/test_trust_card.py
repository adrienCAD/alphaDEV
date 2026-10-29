import json
from alphajet.trust_card import TrustCard


def test_trust_card_roundtrip(tmp_path):
    card = TrustCard(stage="seismic", method="unet", inputs=["f3"], checks={"well_tie": True})
    data = json.loads(card.write(tmp_path).read_text())
    assert data["stage"] == "seismic" and data["checks"]["well_tie"] is True
