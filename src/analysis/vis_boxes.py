import json, sys, cv2, numpy as np
rows = json.load(open(sys.argv[1])); root = sys.argv[2]; out = sys.argv[3]; idxs = [int(i) for i in sys.argv[4].split(',')]
tiles = []
for i in idxs:
    r = rows[i]; img = cv2.imread(f"{root}/{r['img_path']}")
    gtv = cv2.imread(f"{root}/{r['img_path'].replace('images','labels_masks')}", 0)
    img[gtv == 1] = (0.5 * img[gtv == 1] + [0, 0, 127]).astype(np.uint8)
    for c in r['gt_components']:
        x1, y1, x2, y2 = c['box']; cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    for b, s in zip(r['boxes'], r['scores']):
        x1, y1, x2, y2 = map(int, b); cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f'{s:.2f}', (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, f"#{i} IoU={r['iou']*100:.1f} boxes={r['n_boxes']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
    tiles.append(cv2.resize(img, (1024, 512)))
grid = np.vstack([np.hstack(tiles[k:k+2]) for k in range(0, len(tiles), 2)])
cv2.imwrite(out, grid)
