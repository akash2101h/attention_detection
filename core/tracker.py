"""
core/tracker.py – Centroid-based multi-object tracker.
Assigns persistent IDs across frames using greedy nearest-centroid matching.
Handles 20+ students efficiently without external dependencies.
"""

import math
import numpy as np
import config as C


class CentroidTracker:
    def __init__(self):
        self.next_id     = 0
        self.objects     = {}   # id → centroid (cx, cy)
        self.bboxes      = {}   # id → (x1,y1,x2,y2)
        self.disappeared = {}   # id → consecutive missing frames

    # ── public ────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> dict:
        """
        Match new detections to existing tracks.
        Returns {id: {"centroid":(cx,cy), "bbox":(x1,y1,x2,y2)}}
        """
        if not detections:
            self._age_all()
            return self._snapshot()

        new_cents, new_boxes = self._unpack(detections)

        if not self.objects:
            for c, b in zip(new_cents, new_boxes):
                self._register(c, b)
            return self._snapshot()

        self._match(new_cents, new_boxes)
        return self._snapshot()

    # ── private ───────────────────────────────────────────────

    def _register(self, centroid, bbox):
        self.objects[self.next_id]     = centroid
        self.bboxes[self.next_id]      = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid):
        for d in (self.objects, self.bboxes, self.disappeared):
            d.pop(oid, None)

    def _age_all(self):
        for oid in list(self.disappeared):
            self.disappeared[oid] += 1
            if self.disappeared[oid] > C.MAX_DISAPPEARED:
                self._deregister(oid)

    def _unpack(self, detections):
        cents, boxes = [], []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cents.append(((x1 + x2) // 2, (y1 + y2) // 2))
            boxes.append(d["bbox"])
        return cents, boxes

    def _match(self, new_cents, new_boxes):
        obj_ids   = list(self.objects.keys())
        obj_cents = list(self.objects.values())

        # Build distance matrix
        D = np.zeros((len(obj_cents), len(new_cents)))
        for r, oc in enumerate(obj_cents):
            for c, nc in enumerate(new_cents):
                D[r, c] = math.hypot(oc[0] - nc[0], oc[1] - nc[1])

        used_rows, used_cols = set(), set()
        rows, cols = np.unravel_index(np.argsort(D, axis=None), D.shape)

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > C.MAX_DIST:
                break
            oid = obj_ids[r]
            self.objects[oid]     = new_cents[c]
            self.bboxes[oid]      = new_boxes[c]
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        # Unmatched existing tracks → age them
        for r, oid in enumerate(obj_ids):
            if r not in used_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > C.MAX_DISAPPEARED:
                    self._deregister(oid)

        # Unmatched new detections → new tracks
        for c in range(len(new_cents)):
            if c not in used_cols:
                self._register(new_cents[c], new_boxes[c])

    def _snapshot(self):
        return {
            oid: {"centroid": self.objects[oid], "bbox": self.bboxes[oid]}
            for oid in self.objects
        }
