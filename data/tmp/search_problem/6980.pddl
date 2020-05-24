(define (problem blocks-problem)
	(:domain blocksworld-unstack)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0 tower0)
	(:init 
		(arm-empty )
		(clear b2)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(done t0)
		(in-tower t0 tower0)
		(done tower0)
		(tower tower0)
		(on b3 t0)
		(in-tower b3 tower0)
		(done b3)
		(on b4 b3)
		(in-tower b4 tower0)
		(done b4)
		(on b5 b4)
		(in-tower b5 tower0)
		(done b5)
		(on b1 b5)
		(in-tower b1 tower0)
		(done b1)
		(on b0 b1)
		(in-tower b0 tower0)
		(done b0)
		(on-table b2)
		(clear b0)
		(orange b0)
		(pink b8)
		(= (red-count tower0) 0)
		(= (blue-count tower0) 0)
		(= (green-count tower0) 0)
		(= (yellow-count tower0) 0)
		(= (purple-count tower0) 0)
		(= (orange-count tower0) 1)
		(= (pink-count tower0) 0)
	)
	(:goal (and (forall (?x) (done ?x)) (forall (?y) (or (not (orange ?y)) (exists (?x) (and (pink ?x) (on ?x ?y))))) (not (on b2 b0))))
)