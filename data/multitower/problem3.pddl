(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0 t1 tower0 tower1)
	(:init
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0 tower0)
		(clear t0)
		(in-tower t1 tower1)
		(clear t1)
		(blue b0)
		(red b1)
		(orange b2)
		(yellow b3)
		(red b4)
		(blue b5)
		(orange b6)
		(lightyellow b7)
		(yellow b7)
		(purple b8)
		(purple b9)
		(done tower0)
		(done tower1)
		(done t0)
		(done t1)
		(= (blue-count tower1) 0)
		(= (blue-count tower0) 0)
		(= (red-count tower1) 0)
		(= (red-count tower0) 0)
		(= (green-count tower1) 0)
		(= (green-count tower0) 0)
		(= (yellow-count tower1) 0)
		(= (yellow-count tower0) 0)
		(= (purple-count tower1) 0)
		(= (purple-count tower0) 0)
		(= (orange-count tower1) 0)
		(= (orange-count tower0) 0)
		(= (pink-count tower1) 0)
		(= (pink-count tower0) 0)


		(tower tower1)
		(tower tower0)
	)
	(:goal (and
						(forall (?x) (done ?x))
						(forall (?t) (or (not (tower ?t)) (<= (blue-count ?t) 1)))
						(forall (?y) (or (not (red ?y)) (exists (?x) (and (blue ?x) (on ?x ?y)))))
					)
	)
)
