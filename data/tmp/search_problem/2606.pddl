(define (problem blocks-problem)
	(:domain blocksworld-unstack)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0 tower0)
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
		(clear b9)
		(done t0)
		(in-tower t0 tower0)
		(done tower0)
		(tower tower0)
		(on b8 t0)
		(in-tower b8 tower0)
		(done b8)
		(on b7 b8)
		(in-tower b7 tower0)
		(done b7)
		(on-table b9)
		(clear b7)
		(yellow b0)
		(yellow b2)
		(yellow b4)
		(yellow b5)
		(yellow b7)
		(blue b9)
		(= (red-count tower0) 0)
		(= (blue-count tower0) 0)
		(= (green-count tower0) 0)
		(= (yellow-count tower0) 1)
		(= (purple-count tower0) 0)
		(= (orange-count tower0) 0)
		(= (pink-count tower0) 0)
	)
	(:goal (and (forall (?x) (done ?x)) (and (not (on b9 t0)) (not (on b9 b8)) (not (on b9 b7)))))
)