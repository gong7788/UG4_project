(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
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
		(clear b6)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b8 t0)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(blue b0)
		(red b1)
		(blue b1)
		(blue b2)
		(red b3)
		(blue b3)
		(blue b4)
		(blue b5)
		(red b6)
		(blue b6)
		(red b9)
		(blue b9)
		(red b4)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b9 t0)) (not (on b9 b8)) (not (on b9 b7)) (not (on b9 b6)))))
)